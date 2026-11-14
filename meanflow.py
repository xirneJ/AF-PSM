import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.0, c=1e-2):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        mode='i-meanflow',
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # scalar or [low, high] range; None disables CFG
        cfg_scale=[1.0, 5.0],
        cfg_scale_ratio=0.75,
        adaptive_l2_gamma=0.0,
        adaptive_l2_c=1e-2,
    ):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        assert mode in ('meanflow', 'i-meanflow')
        self.mode = mode
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.cfg_scale_ratio = cfg_scale_ratio
        self.use_cfg_w = cfg_scale is not None
        if self.use_cfg_w and isinstance(cfg_scale, (int, float)):
            self.cfg_w_fixed = float(cfg_scale)
            self.cfg_w_range = None
        elif self.use_cfg_w:
            self.cfg_w_fixed = None
            self.cfg_w_range = (float(cfg_scale[0]), float(cfg_scale[1]))
        else:
            self.cfg_w_fixed = None
            self.cfg_w_range = None
        self.adaptive_l2_gamma = adaptive_l2_gamma
        self.adaptive_l2_c = adaptive_l2_c

    def sample_t_r(self, batch_size, device, sample_cfg=True):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)

        if not self.use_cfg_w:
            w = torch.ones(batch_size, device=device)
        elif self.cfg_w_fixed is not None:
            w = torch.full((batch_size,), self.cfg_w_fixed, device=device)
        else:
            low, high = self.cfg_w_range
            if sample_cfg:
                w = torch.ones(batch_size, device=device)
                num = int(self.cfg_scale_ratio * batch_size)
                if num > 0:
                    indices = torch.randperm(batch_size, device=device)[:num]
                    w[indices] = torch.empty(num, device=device).uniform_(low, high)
            else:
                w = torch.full((batch_size,), (low + high) / 2, device=device)

        return t, r, w

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        t, r, w = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x
        v_hat = v

        if c is not None and self.use_cfg_w:
            uncond = torch.ones_like(c) * self.num_classes
            uncond_w = torch.ones_like(w)
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            w = torch.where(cfg_mask, torch.ones_like(w), w)
            w_ = rearrange(w, "b -> b 1 1 1")

            with torch.no_grad():
                _, v_c = model(z, t, r, y=c, w=w)
                _, v_uc = model(z, t, r, y=uncond, w=uncond_w)
                v_hat = v + (1 - 1 / w_) * (v_c - v_uc)
                # cfg_mask_ = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                # v_hat = torch.where(cfg_mask_, v, v_hat)  # redundant: w=1 on uncond => (1 - 1/w_)=0
        else:
            with torch.no_grad():
                _, v_c = model(z, t, r, y=c, w=w)

        with torch.no_grad():
            model_partial = partial(
                model, y=c, w=w, return_v=False, use_flash_attention=False
            )
            _, dudt = torch.autograd.functional.jvp(
                model_partial,
                (z, t, r),
                (v_c, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False,
            )

        u_p, v_p = model(z, t, r, y=c, w=w)

        fm_loss = adaptive_l2_loss(
            v_p - stopgrad(v_hat),
            gamma=self.adaptive_l2_gamma,
            c=self.adaptive_l2_c,
        )

        v_est = u_p + (t_ - r_) * stopgrad(dudt)

        if self.mode == 'meanflow':
            u_tgt = v_hat - (t_ - r_) * dudt
            mf_loss = adaptive_l2_loss(
                u_p - stopgrad(u_tgt),
                gamma=self.adaptive_l2_gamma,
                c=self.adaptive_l2_c,
            )
        else:
            mf_loss = adaptive_l2_loss(
                v_est - stopgrad(v_hat),
                gamma=self.adaptive_l2_gamma,
                c=self.adaptive_l2_c,
            )

        mse_val = {
            'fm_loss': stopgrad(fm_loss),
            'mf_loss': stopgrad(mf_loss),
            'mf_v_mse': (stopgrad(v_est - v_hat) ** 2).mean(),
        }
        return mf_loss + fm_loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda'):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(
            c.shape[0], self.channels, self.image_size, self.image_size, device=device
        )

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        _, _, w = self.sample_t_r(c.shape[0], device, sample_cfg=False)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

            u, _ = model(z, t, r, y=c, w=w)
            z = z - (t_ - r_) * u

        z = self.normer.unnorm(z)
        return z


class AFPSM:
    """
    AF-PSM: Adaptive Frequency-consistent Pixel Split-MeanFlow (JVP-free).

    Conventions are inherited from MeanFlow / iMF:
        z_t = (1 - t) * x + t * e,   e ~ N(0, I)
        v   = e - x                  (constant instantaneous velocity of the linear path)
        t = 1 is noise, t = 0 is data; during training t > r (integrate towards data)

    Design "D" (single clean-image head + closed-form state reconstruction):
      * the network always predicts the clean image x_hat = tanh(...) for any (t, r),
        so the output semantics are consistent across all calls (and tanh is valid);
      * an intermediate sample is reconstructed exactly from the linearity of the
        path, since (x, z_t) and (x, z_s) lie on the same straight line:
            z_s = (1 - s/t) * x_hat + (s/t) * z_t
        (convex weights in [0, 1] because s in [r, t]; no 1/t blow-up);
      * three shared-parameter forwards (no JVP anywhere):
            1) x_hat_p, alpha = model(z_t, t, r)        # direct one-step + split ratio
            2) x_hat_s        = model(z_t, t, s)        # s = alpha*r + (1-alpha)*t
            3) x_hat_r        = model(z_s, s, r)        # z_s reconstructed, fed back
      * four losses: pixel anchoring, frequency consistency, exact midpoint
        supervision and alpha curriculum regularization.

    Note: because the path is linear, the exact midpoint target satisfies
        z_s_true = (1 - s/t) * x + (s/t) * z_t,
    i.e. L_mid is a weighted clean-image loss on the first leg with weight
    (alpha * (t - r) / t)^2. This is why an alpha curriculum that decays to 0.01
    also (deliberately or not) decays the first-leg supervision to ~1e-4.
    """

    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        time_dist=['lognorm', -0.4, 1.0],
        alpha_min=0.01,
        alpha_curriculum='linear',
        alpha_weight=0.1,
        freq_weight=0.5,
        mid_weight=0.5,
        alpha_stopgrad=True,
        detach_xs=False,
        freq_loss='rel_hp',
        freq_gamma=1.0,
        fft_dtype=torch.float32,
        use_flash_attention=False,
    ):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.normer = Normalizer.from_list(normalizer)
        self.time_dist = time_dist

        self.alpha_min = alpha_min
        assert alpha_curriculum in ('linear', 'to_half', 'triangular', 'const', 'none')
        self.alpha_curriculum = alpha_curriculum
        self.alpha_weight = alpha_weight
        self.freq_weight = freq_weight
        self.mid_weight = mid_weight
        self.alpha_stopgrad = alpha_stopgrad
        self.detach_xs = detach_xs
        assert freq_loss in ('rel_hp', 'mse')
        self.freq_loss = freq_loss
        self.freq_gamma = freq_gamma
        self.fft_dtype = fft_dtype
        # memory-efficient attention (SDPA) instead of the O(T^2) naive path;
        # no JVP in AF-PSM, so fused attention kernels are safe to use
        self.use_flash_attention = use_flash_attention

    # ------------------------------------------------------------------ utils
    def sample_t_r(self, batch_size, device):
        """Sample interval endpoints with t > r (integration towards the data end)."""
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        else:
            raise ValueError(f"Unknown time_dist: {self.time_dist[0]}")

        t = torch.tensor(np.maximum(samples[:, 0], samples[:, 1]), device=device)
        r = torch.tensor(np.minimum(samples[:, 0], samples[:, 1]), device=device)
        return t, r

    @staticmethod
    def _state_from_clean(x_hat, z_t, t, u):
        """Reconstruct the state at time u on the line through (t, z_t) and (0, x_hat)."""
        t_ = rearrange(t, "b -> b 1 1 1")
        u_ = rearrange(u, "b -> b 1 1 1")
        t_safe = t_.clamp_min(1e-6)
        return (1 - u_ / t_safe) * x_hat + (u_ / t_safe) * z_t

    def _alpha_target(self, step, total_steps):
        """Curriculum target for the adaptive split ratio (None disables the term)."""
        if self.alpha_curriculum == 'none' or total_steps is None:
            return None
        progress = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
        if self.alpha_curriculum == 'linear':
            return max(self.alpha_min, 1.0 - progress)
        if self.alpha_curriculum == 'to_half':
            return max(0.5, 1.0 - 0.5 * progress)
        if self.alpha_curriculum == 'triangular':
            if progress < 0.5:
                return 1.0 - progress
            return 0.5 + (self.alpha_min - 0.5) * (2.0 * progress - 1.0)
        if self.alpha_curriculum == 'const':
            return 0.5
        raise ValueError(f"Unknown alpha_curriculum: {self.alpha_curriculum}")

    def _radial_weight(self, h, w, device, dtype):
        """Soft high-pass weighting over the rfft2 grid (mean weight 1)."""
        fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(w, device=device, dtype=torch.float32)
        rho = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
        rho = rho / rho.max().clamp_min(1e-8)
        weight = rho ** self.freq_gamma           # gamma=0 -> plain relative MSE
        weight = weight / weight.mean().clamp_min(1e-8)
        return weight.view(1, 1, h, w // 2 + 1).to(dtype)

    def freq_consistency(self, a, b):
        """
        Frequency-domain consistency between two clean predictions.
        rfft2 returns complex tensors of shape (B, C, H, W//2+1); complex tensors
        cannot go through F.mse_loss, so the squared modulus |A-B|^2 is used.
        """
        A = torch.fft.rfft2(a.to(self.fft_dtype), norm='ortho')
        B = torch.fft.rfft2(b.to(self.fft_dtype), norm='ortho')
        diff = (A - B).abs().pow(2)                       # (B, C, H, W//2+1), real
        if self.freq_loss == 'mse':
            return diff.mean()
        w = self._radial_weight(a.shape[-2], a.shape[-1], a.device, diff.dtype)
        num = (w * diff).mean()
        den = (w * A.abs().pow(2)).mean().detach().clamp_min(1e-8)
        return num / den

    # ------------------------------------------------------------------- loss
    def loss(self, model, x, c=None, step=0, total_steps=None):
        """
        AF-PSM training loss (no JVP).
        x: (B, C, H, W) images in [0, 1] (normalized inside)
        c: (B,) class labels (optional)
        step/total_steps: training progress, used by the alpha curriculum
        returns (total_loss, stats_dict)
        """
        if getattr(model, 'output_mode', 'uv') != 'x':
            raise ValueError("AFPSM.loss requires a model with output_mode='x'")

        batch_size, device = x.shape[0], x.device

        t, r = self.sample_t_r(batch_size, device)
        t4 = rearrange(t, "b -> b 1 1 1").detach().clone()
        r4 = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)                            # [-1, 1]
        z_t = (1 - t4) * x + t4 * e                        # (B, C, H, W)
        v = e - x                                          # dz/dt

        # ---- forward 1: direct one-step prediction + adaptive split ratio ----
        x_hat_p, alpha = model(z_t, t, r, y=c,
                               use_flash_attention=self.use_flash_attention)
        alpha_det = alpha.detach()
        alpha4 = rearrange(alpha_det, "b -> b 1 1 1")       # (B, 1, 1, 1)
        # s = alpha * r + (1 - alpha) * t  is the dynamic intermediate point
        alpha_used = alpha if not self.alpha_stopgrad else alpha_det
        s = alpha_used * r + (1 - alpha_used) * t
        s4 = rearrange(s, "b -> b 1 1 1")

        # ---- forward 2: first leg, supervised onto the exact intermediate state
        x_hat_s = model(z_t, t, s, y=c, return_alpha=False,
                        use_flash_attention=self.use_flash_attention)
        t_safe = t4.clamp_min(1e-6)
        z_s_hat = (1 - s4 / t_safe) * x_hat_s + (s4 / t_safe) * z_t
        s_hat = z_s_hat                                     # (B, C, H, W) state at time s

        # ---- forward 3: second leg, starts from the predicted state ----
        z_s_in = s_hat.detach() if self.detach_xs else s_hat
        x_hat_r = model(z_s_in, s, r, y=c, return_alpha=False,
                        use_flash_attention=self.use_flash_attention)

        # ---- exact (detached) targets; sign: z_s = z_t + alpha * (r - t) * v ----
        z_s_true = (z_t + alpha4 * (r4 - t4) * v).detach()

        # ---- 1) pixel anchoring ----
        l_pixel = F.mse_loss(x_hat_p, x) + F.mse_loss(x_hat_r, x)

        # ---- 2) frequency consistency (high-frequency edges of the two clean preds) ----
        l_freq = self.freq_consistency(x_hat_p, x_hat_r)

        # ---- 3) exact midpoint supervision ----
        l_mid = F.mse_loss(s_hat, z_s_true)

        # ---- 4) alpha curriculum regularization ----
        alpha_target = self._alpha_target(step, total_steps)
        if alpha_target is None:
            l_alpha = torch.zeros((), device=device, dtype=l_pixel.dtype)
        else:
            l_alpha = F.mse_loss(alpha, torch.full_like(alpha, float(alpha_target)))

        total = (
            l_pixel
            + self.freq_weight * l_freq
            + self.mid_weight * l_mid
            + self.alpha_weight * l_alpha
        )

        stats = {
            'pixel': l_pixel.detach(),
            'freq': l_freq.detach(),
            'mid': l_mid.detach(),
            'alpha': l_alpha.detach(),
            'alpha_pred': alpha.detach().mean(),
            'alpha_target': torch.tensor(float(alpha_target) if alpha_target is not None else -1.0),
        }
        return total, stats

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=1, device='cuda', adaptive=False):
        """
        x-prediction sampler.
        1 step: x_hat = model(z, 1, 0) is the final image directly.
        K steps: z_{r} = (1 - r/t) * x_hat + (r/t) * z_t   (closed-form state update).
        adaptive=True additionally splits every interval at the predicted alpha.
        """
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        n = c.shape[0]
        z = torch.randn(n, self.channels, self.image_size, self.image_size, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((n,), t_vals[i].item(), device=device)
            r = torch.full((n,), t_vals[i + 1].item(), device=device)
            if adaptive:
                _, alpha = model(z, t, r, y=c,
                                 use_flash_attention=self.use_flash_attention)
                s = alpha * r + (1 - alpha) * t
                x_hat_s = model(z, t, s, y=c, return_alpha=False,
                                use_flash_attention=self.use_flash_attention)
                z = self._state_from_clean(x_hat_s, z, t, s)
                x_hat_r = model(z, s, r, y=c, return_alpha=False,
                                use_flash_attention=self.use_flash_attention)
                z = self._state_from_clean(x_hat_r, z, s, r)
            else:
                x_hat = model(z, t, r, y=c, return_alpha=False,
                              use_flash_attention=self.use_flash_attention)
                z = self._state_from_clean(x_hat, z, t, r)

        return self.normer.unnorm(z)
