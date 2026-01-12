from utils.wandb import fig_to_image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging
import numpy as np
import copy
import wandb
import rootutils
from models.base import BaseGenerativeModel
from metrics.distributed import create_mean_std_rel_err_steps_metric
from metrics.rank0 import create_std_steps_metric

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = logging.getLogger(__name__)


class ConsistencyModel(BaseGenerativeModel):
    """Consistency model for physics simulations."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: nn.Module,
        sigma_min: float = 0.01,
        sigma_max: float = 20.0,
        sigma_data: float = 1.0,
        rho: float = 7.0,
        ema_rate_start: float = 0.99,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        steps_start: int = 8,
        steps_max: int = 128,
        num_epochs: int = 500,
        num_steps_eval: int = 1,
        rand_cons_step: bool = True,
        noise_cond: bool = False,
        noise_schedule: str = "karras",
        consistency_training: str = "ct",
        eval_config: Optional[Dict] = None,
        unroll_steps: int = 1,
        target_loss_weight: float = 0.1,
        slug: Optional[str] = None,
    ):
        """Instantiate the model with explicit init args."""
        super(ConsistencyModel, self).__init__()

        self.net = net

        self.save_hyperparameters(ignore=["net"])
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)

        self.get_noise_schedule = {
            "karras": self._schedule_karras,
            "linear": self._schedule_linear,
        }[noise_schedule]

        self.target_model = self._create_target_model()
        self._initialize_target_model()

        self.register_distributed_metric(
            create_mean_std_rel_err_steps_metric(unbiased=False)
        )
        self.register_rank0_metric(create_std_steps_metric())

    def _create_target_model(self):
        """Create EMA target network."""
        target_model = copy.deepcopy(self.net)
        target_model.eval()

        for param in target_model.parameters():
            param.requires_grad = False

        return target_model

    def _initialize_target_model(self):
        """Init target with online params."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.net.parameters()
            ):
                target_param.data.copy_(online_param.data)

    def _update_target_model(self):
        """EMA update of target."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.net.parameters()
            ):
                target_param.data.mul_(self.ema_rate).add_(
                    online_param.data, alpha=1 - self.ema_rate
                )

    def _schedule_karras(self, num_steps: int):
        """Karras noise schedule."""
        step_indices = torch.arange(num_steps + 1, dtype=torch.float32)

        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)

        t = (
            sigma_min_rho + step_indices / num_steps * (sigma_max_rho - sigma_min_rho)
        ) ** self.rho
        return t

    def _schedule_linear(self, num_steps: int):
        """Linear schedule."""
        step_indices = torch.arange(num_steps + 1, dtype=torch.float32)
        t = self.sigma_min + (self.sigma_max - self.sigma_min) * step_indices / (
            num_steps
        )
        t = torch.clamp(t, min=self.sigma_min, max=self.sigma_max)
        return t

    def _update_ema_and_steps(self):
        """Steps at current training iteration."""

        k, K = self.trainer.current_epoch, self.num_epochs
        s0, s1 = self.steps_start, self.steps_max
        mu0 = self.ema_rate_start

        # formulas from CM paper
        N = np.ceil(np.sqrt(k / K * ((s1 + 1) ** 2 - s0**2) + s0**2) - 1) + 1
        mu = np.exp(s0 * np.log(mu0) / N)

        self.num_steps = min(int(N), self.steps_max)
        self.ema_rate = mu

    def c_skip(self, sigma):
        """Skip-scale."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Output-scale."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma):
        """Input-scale."""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        """Noise-scale."""
        # sigma_clamped = torch.clamp(sigma, min=self.sigma_min)
        return 0.25 * torch.log(sigma)

    def consistency_function(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        use_target: bool = False,
    ):
        """Denoise x at noise sigma with cond.

        x: [B, C, H, W]
        sigma: [B]
        cond: [B, S*C, H, W]
        """
        c_in = self.c_in(sigma)
        c_out = self.c_out(sigma)
        c_skip = self.c_skip(sigma)

        x_in = torch.cat([cond, x * c_in[:, None, None, None]], dim=1)

        model = self.target_model if use_target else self.net
        f_theta = model(x_in, self.c_noise(sigma))

        return x * c_skip[:, None, None, None] + f_theta * c_out[:, None, None, None]

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Compute training loss with unrolling.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert target.ndim == 5, f"target: [B,1,C,H,W], got {tuple(target.shape)}"
        assert target.shape[1] == 1, f"target dim 1 must be 1, got {target.shape[1]}"
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"

        B = target.shape[0]
        U = self.unroll_steps
        T = self.num_steps
        device = target.device

        # [B, S, C, H, W] -> [B, S*C, H, W]
        cond = cond.flatten(1, 2)
        sigmas = self.get_noise_schedule(T).to(device)

        t = torch.randint(1, T + 1, (B,), device=device)
        
        x_est = target[:, 0]
        total_loss = 0.0

        for _ in range(U):
            active = t > 0
            if not active.any():
                break

            t_curr = t
            t_prev = (t - 1).clamp(min=0)

            sig_t = sigmas[t_curr]
            sig_prev = sigmas[t_prev]

            z = torch.randn_like(x_est)
            x_t = x_est + sig_t[:, None, None, None] * z
            x_prev = x_est + sig_prev[:, None, None, None] * z

            with torch.no_grad():
                f_teacher = self.consistency_function(
                    x_prev, sig_prev, cond, use_target=True
                )

            f_student = self.consistency_function(x_t, sig_t, cond, use_target=False)

            loss = torch.nn.functional.mse_loss(f_student[active], f_teacher[active])
            total_loss += loss

            x_est = f_student.detach()
            t -= 1

        return total_loss / U

    def generate_samples(
        self,
        cond: torch.Tensor,
        num_steps: int,
        use_ema: bool = True,
        requires_grad: bool = False,
    ):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert num_steps >= 1, "num_steps >= 1"

        device = cond.device
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        T = self.num_steps
        t_schedule = self.get_noise_schedule(T).to(device)

        indices = torch.linspace(T, 0, num_steps, dtype=torch.long, device=device)

        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            x = torch.randn(B, C, H, W, device=device) * self.sigma_max

            for i, idx in enumerate(indices):
                sigma = t_schedule[idx]

                x0 = self.consistency_function(
                    x, sigma.view(1).repeat(B), cond_flat, use_target=use_ema
                )

                if i == len(indices) - 1:
                    return x0.unsqueeze(1)

                next_sigma = t_schedule[indices[i + 1]]

                term = next_sigma**2 - self.sigma_min**2
                term = torch.maximum(term, torch.zeros_like(term))

                x = x0 + torch.randn_like(x0) * torch.sqrt(term)

            return x0.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        """Log training loss and stepsize."""
        loss = super().training_step(batch, batch_idx)
        # self.log_dict was previously calling the custom log_dict
        # Now we use self.log (Lightning)
        self.log(
            "train/num_steps", self.num_steps, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True
        )
        self.log(
            "train/ema_rate", self.ema_rate, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA target network after optimizer step."""
        self._update_target_model()

    def on_train_epoch_start(self):
        self._update_ema_and_steps()

    def setup(self, stage):
        self._update_ema_and_steps()
