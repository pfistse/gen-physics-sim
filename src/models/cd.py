import copy
from typing import Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig

from models.base import BaseGenerativeModel
from metrics.distributed import (
    create_cm_cons_err_gap_metric,
    create_cm_enstr_err_ssched_metric,
    create_cm_mean_std_err_ssched_metric,
    create_enstr_err_steps_metric,
    create_mean_std_err_steps_metric,
    create_cons_err_steps_metric,
)
from metrics.rank0 import create_std_steps_metric


class ConsistencyDistillationModel(BaseGenerativeModel):
    """Consistency distillation with EDM teacher."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: DictConfig,
        pretrained_config: DictConfig,
        ckpt_path: str,
        ema_rate: float = 0.999,
        num_student_steps: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps_eval: int = 1,
        eval_config: Optional[Dict] = None,
        boundary_loss_weight: float = 0.1, #0.1,
        snr_weighting: bool = True,
    ):
        super().__init__()

        net = hydra.utils.instantiate(net)

        self.save_hyperparameters(ignore=["net"])
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)
        self.num_steps = int(self.num_student_steps)

        self.teacher = hydra.utils.instantiate(pretrained_config)
        self._load_weights(self.teacher, ckpt_path)
        self.teacher.eval().requires_grad_(False)
        self.net.load_state_dict(self.teacher.net.state_dict(), strict=True)

        self.sigma_min = float(self.teacher.sigma_min)
        self.sigma_max = float(self.teacher.sigma_max)
        self.sigma_data = float(self.teacher.sigma_data)
        self.rho = float(self.teacher.rho)

        assert self.num_steps >= 2, "num_steps >= 2"

        self._build_student_schedule()
        self.target_model = self._create_target_model()
        self._initialize_target_model()

        self.register_distributed_metric(
            create_mean_std_err_steps_metric(unbiased=False)
        )
        self.register_distributed_metric(create_enstr_err_steps_metric(unbiased=False))
        self.register_distributed_metric(create_cons_err_steps_metric(unbiased=False))
        self.register_distributed_metric(create_cm_mean_std_err_ssched_metric())
        self.register_distributed_metric(create_cm_enstr_err_ssched_metric())
        self.register_distributed_metric(create_cm_cons_err_gap_metric())
        self.register_rank0_metric(create_std_steps_metric())

    def _load_weights(self, model, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)

    def _build_student_schedule(self) -> None:
        step_indices = torch.arange(self.num_steps, dtype=torch.float32)
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (
            sigma_max_rho
            + step_indices / (self.num_steps - 1) * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
        self.register_buffer("sigmas", sigmas)

    def _sigma_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        step_indices = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (
            sigma_max_rho
            + step_indices / num_steps * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
        sigmas[-1] = 0.0
        return sigmas

    def _create_target_model(self):
        target_model = copy.deepcopy(self.net)
        target_model.eval()
        for param in target_model.parameters():
            param.requires_grad = False
        return target_model

    def _initialize_target_model(self):
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.net.parameters()
            ):
                target_param.data.copy_(online_param.data)

    def _update_target_model(self):
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.net.parameters()
            ):
                target_param.data.mul_(self.ema_rate).add_(
                    online_param.data, alpha=1 - self.ema_rate
                )

    def c_skip(self, sigma):
        s = sigma - self.sigma_min
        return self.sigma_data**2 / (s**2 + self.sigma_data**2)

    def c_out(self, sigma):
        s = sigma - self.sigma_min
        return self.sigma_data * s / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input-scale."""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Noise-scale."""
        return 0.25 * torch.log(sigma)

    def consistency_function(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        """Denoise x at noise sigma with cond.

        x: [B, C, H, W]
        sigma: [B]
        cond: [B, S*C, H, W]
        """
        assert x.ndim == 4, f"x: [B,C,H,W], got {tuple(x.shape)}"
        assert sigma.ndim == 1, f"sigma: [B], got {tuple(sigma.shape)}"
        assert cond.ndim == 4, f"cond: [B,S*C,H,W], got {tuple(cond.shape)}"
        assert x.shape[0] == sigma.shape[0] == cond.shape[0], "batch size mismatch"

        c_in = self.c_in(sigma)
        c_out = self.c_out(sigma)
        c_skip = self.c_skip(sigma)

        x_in = torch.cat([cond, x * c_in[:, None, None, None]], dim=1)
        model = self.target_model if use_target else self.net
        f_theta = model(x_in, self.c_noise(sigma))[:, -x.shape[1] :, :, :]
        return x * c_skip[:, None, None, None] + f_theta * c_out[:, None, None, None]

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Compute distillation loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert target.ndim == 5, f"target: [B,1,C,H,W], got {tuple(target.shape)}"
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert target.size(1) == 1, "target second dim == 1"
        assert self.num_steps >= 2, "num_steps >= 2"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x0 = target.squeeze(1)

        t_idx = torch.randint(
            0, self.num_steps - 1, (B,), device=target.device
        )
        t_next = t_idx + 1

        sigma_t = self.sigmas[t_idx].to(dtype=x0.dtype)
        sigma_next = self.sigmas[t_next].to(dtype=x0.dtype)

        noise = torch.randn_like(x0)
        x = x0 + sigma_t[:, None, None, None] * noise

        with torch.no_grad():
            x0_pred = self.teacher.denoise(x, sigma_t, cond_flat)
            eps_pred = (x - x0_pred) / sigma_t[:, None, None, None]
            x_next = x0_pred + sigma_next[:, None, None, None] * eps_pred
            target_pred = self.consistency_function(
                x_next,
                sigma_next,
                cond_flat,
                use_target=True,
            )

        student_pred = self.consistency_function(
            x,
            sigma_t,
            cond_flat,
            use_target=False,
        )

        boundary_pred = None
        if self.boundary_loss_weight > 0:
            sigma_min = torch.full(
                (B,), self.sigma_min, device=x0.device, dtype=x0.dtype
            )
            x_min = x0 + sigma_min[:, None, None, None] * torch.randn_like(x0)
            boundary_pred = self.consistency_function(
                x_min,
                sigma_min,
                cond_flat,
                use_target=False,
            )

        if self.snr_weighting:
            mse = (student_pred - target_pred).square().mean(dim=(1, 2, 3))
            weight = (sigma_t**2 + self.sigma_data**2) / (
                (sigma_t * self.sigma_data) ** 2
            )
            loss = (mse * weight).mean()
        else:
            loss = (student_pred - target_pred).square().mean()

        if boundary_pred is not None:
            boundary_loss = (boundary_pred - x0).square().mean()
            loss = loss + self.boundary_loss_weight * boundary_loss

        return loss

    @torch.no_grad()
    def generate_samples(
        self,
        cond: torch.Tensor,
        num_steps: int,
        use_ema: bool = True,
    ):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert num_steps >= 1, "num_steps >= 1"

        device = cond.device
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        indices = torch.linspace(
            0, self.num_steps - 1, num_steps + 1, device=device
        )[:-1].round().to(torch.long)
        sigmas = self.sigmas[indices].to(device)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        x = torch.randn(B, C, H, W, device=device, dtype=cond.dtype) * sigmas[0]

        for i in range(num_steps):
            sigma = sigmas[i].expand(B)
            x = self.consistency_function(x, sigma, cond_flat, use_target=use_ema)

        if i < num_steps - 1:
            sigma_next = sigmas[i + 1]
            noise_std = torch.sqrt(torch.clamp(sigma_next**2 - self.sigma_min**2, min=0.0))
            x = x + noise_std * torch.randn_like(x)


        return x.unsqueeze(1)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_target_model()
