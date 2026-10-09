"""
Linear Noise Scheduler for DDPM
================================
Implements:
  • Linear β schedule over T timesteps.
  • Forward (diffusion) process:  q(x_t | x_0)  — closed-form.
  • Reverse (denoising) step:     p(x_{t-1} | x_t, ε_θ)
  • DDIM-style deterministic sampling (optional, for fast inference).

Reference:
  Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020).
  https://arxiv.org/abs/2006.11239
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor


class LinearNoiseScheduler:
    """
    Linear β noise schedule for DDPM.

    Parameters
    ----------
    num_timesteps : int
        Total diffusion steps T (default 1000).
    beta_start : float
        β₁ — noise variance at the first step.
    beta_end : float
        β_T — noise variance at the last step.
    device : str | torch.device
        Device to store schedule tensors on.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start:    float = 1e-4,
        beta_end:      float = 2e-2,
        device: str | torch.device = "cpu",
    ):
        self.T      = num_timesteps
        self.device = torch.device(device)

        # ── β schedule ──────────────────────────────────────────────────
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps,
                                    dtype=torch.float64).to(self.device)

        # ── Derived quantities ───────────────────────────────────────────
        self.alphas          = 1.0 - self.betas
        self.alphas_cumprod  = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1}  –  prepend 1.0 so index 0 gives ᾱ₀ = 1
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # ── Forward process coefficients ─────────────────────────────────
        self.sqrt_alphas_cumprod          = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

        # ── Posterior q(x_{t-1} | x_t, x_0) ────────────────────────────
        # σ²_q = β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # log clipped to avoid log(0) at t=0
        self.log_posterior_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )

        # ── Posterior mean coefficients ──────────────────────────────────
        # μ̃_t = coeff1 * x_0 + coeff2 * x_t
        self.posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt()
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * self.alphas.sqrt()
            / (1.0 - self.alphas_cumprod)
        )

    # ─────────────────────────────────────────────────────────────────────
    #  INDEXING HELPER
    # ─────────────────────────────────────────────────────────────────────

    def _extract(self, a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
        """
        Extract scalar values at timesteps ``t`` and broadcast to ``x_shape``.
        ``a`` must be a 1-D tensor of length T; ``t`` is a 1-D integer batch tensor.
        """
        batch_size = t.shape[0]
        out = a.gather(0, t.to(self.device))                     # (B,)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # (B,1,1,1)

    # ─────────────────────────────────────────────────────────────────────
    #  FORWARD PROCESS  q(x_t | x_0)
    # ─────────────────────────────────────────────────────────────────────

    def q_sample(
        self,
        x_start: Tensor,
        t:       Tensor,
        noise:   Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample x_t from q(x_t | x_0) in a single forward pass.

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

        Parameters
        ----------
        x_start : (B, C, H, W)  clean images in [-1, 1]
        t       : (B,)          integer timesteps
        noise   : (B, C, H, W)  optional pre-sampled noise

        Returns
        -------
        x_t    : (B, C, H, W)  noisy image at step t
        noise  : (B, C, H, W)  the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus * noise
        return x_t, noise

    # ─────────────────────────────────────────────────────────────────────
    #  REVERSE STEP  p(x_{t-1} | x_t, ε_θ)
    # ─────────────────────────────────────────────────────────────────────

    def p_mean_variance(
        self,
        x_t:           Tensor,
        t:             Tensor,
        predicted_noise: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute posterior mean and variance for the reverse step.

        Returns (model_mean, posterior_variance, log_posterior_variance_clipped).
        """
        # Predict x_0 from x_t and ε_θ
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        x_recon = (x_t - sqrt_one_minus * predicted_noise) / sqrt_alpha_bar
        x_recon = x_recon.clamp(-1.0, 1.0)

        # Posterior mean  μ̃_t(x_t, x_0)
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        model_mean = coef1 * x_recon + coef2 * x_t

        # Posterior variance  σ²_t
        posterior_var     = self._extract(self.posterior_variance, t, x_t.shape)
        log_posterior_var = self._extract(
            self.log_posterior_variance_clipped, t, x_t.shape
        )
        return model_mean, posterior_var, log_posterior_var

    @torch.no_grad()
    def p_sample(
        self,
        model_output: Tensor,
        x_t:          Tensor,
        t:            Tensor,
    ) -> Tensor:
        """
        Single DDPM reverse-diffusion step.
        Returns x_{t-1}.
        """
        mean, _, log_var = self.p_mean_variance(x_t, t, model_output)
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (x_t.ndim - 1)))
        x_prev = mean + nonzero_mask * (0.5 * log_var).exp() * noise
        return x_prev

    # ─────────────────────────────────────────────────────────────────────
    #  DDIM DETERMINISTIC SAMPLING (faster inference)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_step(
        self,
        model_output: Tensor,
        x_t:          Tensor,
        t:            Tensor,
        t_prev:       Tensor,
        eta:          float = 0.0,    # 0 = fully deterministic DDIM
    ) -> Tensor:
        """
        DDIM reverse step (Song et al., 2020).
        ``eta=0`` → deterministic;  ``eta=1`` → recovers DDPM.
        """
        alpha_bar      = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev.clamp(min=0), x_t.shape)
        sqrt_one_minus = (1.0 - alpha_bar).sqrt()

        # Predict x_0
        x0_pred = (x_t - sqrt_one_minus * model_output) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Direction pointing to x_t
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt() * \
                (1 - alpha_bar / alpha_bar_prev).sqrt()

        dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * model_output
        noise  = sigma * torch.randn_like(x_t)

        x_prev = alpha_bar_prev.sqrt() * x0_pred + dir_xt + noise
        return x_prev

    # ─────────────────────────────────────────────────────────────────────
    #  LOSS
    # ─────────────────────────────────────────────────────────────────────

    def training_loss(
        self,
        predicted_noise: Tensor,
        target_noise:    Tensor,
    ) -> Tensor:
        """
        Simple L2 loss between predicted and actual noise (ε_θ − ε).
        This is Eq. 14 from Ho et al. 2020.
        """
        return F.mse_loss(predicted_noise, target_noise)

    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LinearNoiseScheduler(T={self.T}, "
            f"β=[{self.betas[0]:.1e} → {self.betas[-1]:.1e}])"
        )
