import copy
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
import hydra
from omegaconf import DictConfig
from models.base import BaseGenerativeModel


class ADDFlowMatching(BaseGenerativeModel):
    """Adversarial Diffusion Distillation (ADD) for Flow Matching."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        pretrained_config: DictConfig,
        discriminator_config: DictConfig,
        ckpt_path: str,
        num_steps_eval: int,
        lr_g: float = 1e-5,
        lr_d: float = 2e-5,
        b1_g: float = 0.9,
        b2_g: float = 0.99,
        b1_d: float = 0.5,
        b2_d: float = 0.999,
        gen_rate: int = 10,
        weight_decay: float = 1e-4,
        num_student_steps: int = 4,
        num_teacher_steps: int = 10,
        g_loss_lambda: float = 0.5,
        r1_gamma: float = 1.0,
        eval_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)

        self.automatic_optimization = False

        self.teacher = hydra.utils.instantiate(pretrained_config)
        self._load_weights(self.teacher, ckpt_path)
        self.teacher.eval().requires_grad_(False)

        self.student = copy.deepcopy(self.teacher)
        self.student.train().requires_grad_(True)

        self.discriminator = hydra.utils.instantiate(
            discriminator_config, original_unet=self.student.net
        )

    def _load_weights(self, model, path):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)

    def _solve_euler(self, model, x, cond_flat, steps):
        """Euler integration loop."""
        dt = 1.0 / steps
        t = torch.tensor(0.0, device=x.device)
        B = x.shape[0]

        for _ in range(steps):
            t_vec = torch.full((B,), t.item(), device=x.device)
            v_pred = model.net(torch.cat([cond_flat, x], dim=1), t_vec)
            x = x + v_pred * dt
            t = t + dt
        return x

    def student_forward(self, cond, x_init=None):
        """Forward pass for student (1 step usually).

        cond: [B, S, C, H, W]
        x_init: [B, C, H, W] optional noise
        """
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        if x_init is None:
            x_init = torch.randn(B, C, H, W, device=cond.device)

        return self._solve_euler(
            self.student, x_init, cond_flat, self.num_student_steps
        )

    def forward(self, cond, num_steps=None, **kwargs):
        return self.student_forward(cond).unsqueeze(1)

    def training_step(self, batch, batch_idx):
        cond, target = batch
        cond, target = cond.detach(), target.detach()

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x_real = target.squeeze(1)

        opt_g, opt_d = self.optimizers()
        t_zero = torch.zeros((B,), device=self.device, dtype=torch.long)

        noise = torch.randn(B, C, H, W, device=self.device)

        # train discriminator
        self.toggle_optimizer(opt_d)
        x_real.requires_grad = True

        logits_real_dict = self.discriminator(
            torch.cat([cond_flat, x_real], dim=1), t_zero
        )

        with torch.no_grad():
            x_fake = self.student_forward(cond, x_init=noise)

        logits_fake_dict = self.discriminator(
            torch.cat([cond_flat, x_fake.detach()], dim=1), t_zero
        )

        loss_d_hinge = 0.0
        for key in logits_real_dict.keys():
            real_score = logits_real_dict[key]
            fake_score = logits_fake_dict[key]

            self.log(f"scores/{key}/real", real_score.mean(), on_step=True)
            self.log(f"scores/{key}/fake", fake_score.mean(), on_step=True)

            l_real = F.relu(1.0 - real_score).mean()
            l_fake = F.relu(1.0 + fake_score).mean()
            loss_d_hinge += l_real + l_fake

        all_real_logits_sum = sum([l.sum() for l in logits_real_dict.values()])

        grad_real = torch.autograd.grad(
            outputs=all_real_logits_sum,
            inputs=x_real,
            create_graph=True,
            retain_graph=True,
        )[0]
        r1_penalty = grad_real.pow(2).view(B, -1).sum(1).mean()

        loss_d = loss_d_hinge + (self.hparams.r1_gamma / 2.0) * r1_penalty

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        x_real.requires_grad = False
        self.log_dict(
            {"train/d_loss": loss_d, "train/d_r1": r1_penalty},
            on_step=True,
            on_epoch=True,
        )

        # train student
        if (batch_idx + 1) % self.hparams.gen_rate == 0:
            self.toggle_optimizer(opt_g)

            x_fake = self.student_forward(cond, x_init=noise)

            logits_fake_g_dict = self.discriminator(
                torch.cat([cond_flat, x_fake], dim=1), t_zero
            )

            loss_adv = 0.0
            for score in logits_fake_g_dict.values():
                loss_adv += -score.mean()

            with torch.no_grad():
                x_teacher = self._solve_euler(
                    self.teacher, noise.clone(), cond_flat, self.num_teacher_steps
                )

            indices = torch.randint(0, self.num_student_steps, (B,), device=self.device)
            t = indices.float() / self.num_student_steps
            t_exp = t.view(B, 1, 1, 1)
            x_t = (1 - t_exp) * noise + t_exp * x_teacher
            v_pred = self.student.net(torch.cat([cond_flat, x_t], dim=1), t)
            v_target = x_teacher - noise
            loss_distill = F.mse_loss(v_pred, v_target)

            loss_g = (1 - self.hparams.g_loss_lambda) * loss_adv + (
                self.hparams.g_loss_lambda * loss_distill
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()
            self.untoggle_optimizer(opt_g)

            self.log_dict(
                {
                    "train/g_loss": loss_g,
                    "train/g_adv": loss_adv,
                    "train/g_distill": loss_distill,
                },
                on_step=True,
                on_epoch=True,
            )

    def generate_samples(
        self, cond: torch.Tensor, num_steps: Optional[int] = None, **kwargs
    ):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        with torch.no_grad():
            sample = self.student_forward(cond)
        return sample.unsqueeze(1)

    def generate_samples(
        self, cond: torch.Tensor, num_steps: Optional[int] = None, **kwargs
    ):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        with torch.no_grad():
            sample = self.student_forward(cond)
        return sample.unsqueeze(1)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.hparams.lr_g,
            betas=(self.hparams.b1_g, self.hparams.b2_g),
            weight_decay=self.hparams.weight_decay,
        )

        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.lr_d,
            betas=(self.hparams.b1_d, self.hparams.b2_d),
            weight_decay=self.hparams.weight_decay,
        )
        return [opt_g, opt_d]
