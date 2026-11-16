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
        # ------------------------------------------------------------------
        # AF-PSM v2 knobs.  The defaults reproduce the v1 objective exactly,
        # so existing configs keep working unchanged.
        # ------------------------------------------------------------------
        alpha_mode='predicted',
        split_min=0.1,
        pixel_weight=1.0,
        anchor_weight=1.0,
        cons_weight=0.0,
        cfg_ratio=0.0,
        cfg_scale=None,
        cfg_scale_ratio=0.75,
        sample_w=None,
        # ------------------------------------------------------------------
        # v3: progressive distillation from a frozen multi-step teacher.
        # When train.py attaches a teacher (--teacher_ckpt), the loss target
        # becomes the teacher's K-step rollout at (t, r) instead of the
        # self-referential CFG target.  The 1-step student then learns a real
        # sample (not the class conditional mean), which is what makes 1-NFE
        # work on natural images.
        # ------------------------------------------------------------------
        teacher_steps=2,
        teacher_w=1.0,
        endpoint_prob=0.0,
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

        # ---- v2: how the split point s is chosen --------------------------
        # 'predicted' = alpha head + curriculum (v1)
        # 'random'    = s between r and t with a random fraction (every split is
        #               an equally valid training interval, so a curriculum has no
        #               signal source -- see the v1 analysis)
        assert alpha_mode in ('predicted', 'random')
        self.alpha_mode = alpha_mode
        self.split_min = float(split_min)
        # ---- v2: which branch is anchored to the target and which is distilled
        # v1 used pixel_weight=anchor_weight=1, cons_weight=0 (both branches -> x).
        # v2 uses pixel_weight=0, anchor_weight=1, cons_weight=1: only the composed
        # path is anchored to data; the direct path distils it (M1).
        self.pixel_weight = pixel_weight
        self.anchor_weight = anchor_weight
        self.cons_weight = cons_weight
        # ---- v2: CFG distillation, same recipe as the iMF baseline ---------
        # w is a model input; the guided target is built from no-grad conditional
        # / unconditional predictions, so sampling needs only one forward pass.
        self.cfg_ratio = float(cfg_ratio)
        self.cfg_scale_ratio = float(cfg_scale_ratio)
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
        # explicit sampling-time guidance scale (None -> midpoint of cfg_scale,
        # mirroring MeanFlow.sample_each_class)
        self.sample_w_value = None if sample_w is None else float(sample_w)

        # ---- v3: frozen teacher (attached by train.py via attach_teacher) ----
        self.teacher_steps = int(teacher_steps)
        self.teacher_w = float(teacher_w)
        # fraction of each batch forced onto the inference endpoint (t=1, r=0);
        # a 1-NFE student is only as good as its coverage of that point
        self.endpoint_prob = float(endpoint_prob)
        self.teacher = None
        self.teacher_output_mode = None
        self.teacher_mode = None

    # ------------------------------------------------------------------ utils
    def sample_t_r(self, batch_size, device):
        """Sample interval endpoints with t > r (integration towards the data end)."""
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        elif self.time_dist[0] == 'mixed':
            # ['mixed', mu, sigma, p_uniform]: elementwise mix of logit-normal and
            # uniform.  The uniform part is what puts mass on the *inference*
            # endpoints (t ~ 1, r ~ 0), which the logit-normal almost never visits
            # (P(t > 0.99) ~ 6e-7 for the default mu=-0.4, sigma=1.0).
            mu, sigma, p_uniform = self.time_dist[-3], self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            lognorm = 1.0 / (1.0 + np.exp(-normal_samples))
            uniform = np.random.rand(batch_size, 2).astype(np.float32)
            pick_uniform = np.random.rand(batch_size, 2) < float(p_uniform)
            samples = np.where(pick_uniform, uniform, lognorm).astype(np.float32)
        else:
            raise ValueError(f"Unknown time_dist: {self.time_dist[0]}")

        t = torch.tensor(np.maximum(samples[:, 0], samples[:, 1]), device=device)
        r = torch.tensor(np.minimum(samples[:, 0], samples[:, 1]), device=device)
        return t, r

    def _sample_w_batch(self, batch_size, device):
        """Training-time guidance scales (same distribution as the iMF baseline)."""
        if not self.use_cfg_w:
            return torch.ones(batch_size, device=device)
        if self.cfg_w_fixed is not None:
            return torch.full((batch_size,), self.cfg_w_fixed, device=device)
        low, high = self.cfg_w_range
        w = torch.ones(batch_size, device=device)
        num = int(self.cfg_scale_ratio * batch_size)
        if num > 0:
            idx = torch.randperm(batch_size, device=device)[:num]
            w[idx] = torch.empty(num, device=device).uniform_(low, high)
        return w

    def _sampling_w(self, batch_size, device):
        """Guidance scale used at sampling time (one forward pass, no 2x NFE).

        Precedence: an explicitly fixed cfg_scale (e.g. ``fid_eval --cfg_scale 2.0``)
        wins over the config default ``sample_w``, which in turn wins over the
        midpoint of the training ``cfg_scale`` range.  ``--no_cfg`` disables the
        guidance conditioning entirely, so sampling always uses w = 1.
        """
        if not self.use_cfg_w:
            return torch.ones(batch_size, device=device)
        if self.cfg_w_fixed is not None:
            return torch.full((batch_size,), self.cfg_w_fixed, device=device)
        if self.sample_w_value is not None:
            return torch.full((batch_size,), self.sample_w_value, device=device)
        low, high = self.cfg_w_range
        return torch.full((batch_size,), (low + high) / 2.0, device=device)

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

    # ------------------------------------------------------- teacher (v3)
    def attach_teacher(self, model, output_mode='uv', mode='i-meanflow'):
        """Attach a frozen multi-step teacher used as the distillation target.

        The teacher is *not* part of the student's optimizer: it is put in eval
        mode and every parameter is frozen.  ``output_mode`` selects how the
        teacher's own sampler is driven:
            'x'  -> closed-form DDIM state update (an AF-PSM style teacher)
            'uv' -> velocity update z_r = z_t - (t - r) * u (an iMF teacher)
        """
        device = next(model.parameters()).device
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.teacher = model
        self.teacher_output_mode = output_mode
        self.teacher_mode = mode
        return model

    @torch.no_grad()
    def _teacher_clean(self, z_t, t, r, c=None):
        """Roll the frozen teacher K steps from (z_t, t) to r, return clean x.

        Both parametrisations are supported:
          * output_mode == 'x' : the teacher predicts the clean image directly;
            the state is reconstructed with the closed-form DDIM update.
          * output_mode == 'uv': the teacher predicts (u, v).  For the linear
            interpolant v is constant along the path, so the final state can be
            converted back to a clean image with  x = z_r - r * u.
        The result is detached: it is a target, not a differentiable path.
        """
        teacher = self.teacher
        assert teacher is not None, 'no teacher attached (call attach_teacher)'
        K = max(int(self.teacher_steps), 1)
        device = z_t.device
        w = torch.full((z_t.shape[0],), self.teacher_w, device=device)
        r4 = rearrange(r, "b -> b 1 1 1")

        z, tt = z_t, t
        last_clean, u_last = None, None
        for k in range(K):
            t_next = r + (t - r) * (K - 1 - k) / K
            t4 = rearrange(tt, "b -> b 1 1 1")
            n4 = rearrange(t_next, "b -> b 1 1 1")
            if self.teacher_output_mode == 'x':
                x_hat = teacher(z, tt, t_next, y=c, w=w, return_alpha=False,
                                use_flash_attention=False)
                last_clean = x_hat
                z = ((1 - n4 / t4.clamp_min(1e-6)) * x_hat
                     + (n4 / t4.clamp_min(1e-6)) * z)
            else:
                u, v = teacher(z, tt, t_next, y=c, w=w, return_v=True,
                               use_flash_attention=False)
                u_last = u
                z = z - (t4 - n4) * u
                last_clean = z - r4 * u          # x = z_r - r * u (u == v here)
            tt = t_next
        return last_clean.detach()

    # ------------------------------------------------------------------- loss
    def loss(self, model, x, c=None, step=0, total_steps=None):
        """
        AF-PSM training loss (no JVP).

        v1 defaults reproduce the original objective:
            L = MSE(x_p, x) + MSE(x_r, x) + freq/mid/alpha terms
        v2 (see configs/afpsm_*_v2.py) uses the consistency formulation:
            L = anchor_weight * MSE(x_r, T) + cons_weight * MSE(x_p, sg(x_r))
        where x_r is the two-leg composition (anchored to the data-side target T)
        and x_p is the direct prediction that distils it (M1); T is the CFG-guided
        target when CFG distillation is enabled (M3).

        x: (B, C, H, W) images in [0, 1] (normalized inside)
        c: (B,) class labels (optional)
        step/total_steps: training progress, only used by the v1 alpha curriculum
        returns (total_loss, stats_dict)
        """
        if getattr(model, 'output_mode', 'uv') != 'x':
            raise ValueError("AFPSM.loss requires a model with output_mode='x'")

        batch_size, device = x.shape[0], x.device

        t, r = self.sample_t_r(batch_size, device)
        # v3: force part of the batch onto the 1-NFE inference endpoint (t=1, r=0).
        # The logit-normal / uniform mixture only visits it with small probability,
        # but a 1-step student is judged exactly there.
        if self.endpoint_prob > 0:
            ep = torch.rand(batch_size, device=device) < self.endpoint_prob
            t = torch.where(ep, torch.ones_like(t), t)
            r = torch.where(ep, torch.zeros_like(r), r)
        t4 = rearrange(t, "b -> b 1 1 1").detach().clone()
        r4 = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)                            # [-1, 1]
        z_t = (1 - t4) * x + t4 * e                        # (B, C, H, W)
        v = e - x                                          # dz/dt

        # ---- conditioning: label dropout + CFG scale (M3) ------------------
        # Mirrors the iMF baseline: w is a model input and the guided target is
        # built from no-grad conditional / unconditional predictions, so sampling
        # stays at a single forward pass (no doubled NFE).
        if self.teacher is not None:
            # ---- v3: the frozen multi-step teacher provides the target --------
            # No self-referential CFG target: the target is a teacher rollout, so
            # the student learns to reproduce a real sample instead of the class
            # conditional mean (which is what killed 1-NFE on CIFAR in v2).
            c_cond, w_cond = c, None
            target = self._teacher_clean(z_t, t, r, c=c)
        elif c is not None and self.use_cfg_w:
            uncond = torch.full_like(c, self.num_classes)
            w = self._sample_w_batch(batch_size, device)
            drop = torch.rand(batch_size, device=device) < self.cfg_ratio
            c_cond = torch.where(drop, uncond, c)
            w_cond = torch.where(drop, torch.ones_like(w), w)
            with torch.no_grad():
                x_c = model(z_t, t, r, y=c_cond, w=w_cond, return_alpha=False,
                            use_flash_attention=self.use_flash_attention)
                x_uc = model(z_t, t, r, y=uncond, w=torch.ones_like(w),
                             return_alpha=False,
                             use_flash_attention=self.use_flash_attention)
            # x-space equivalent of iMF's v_hat = v + (1 - 1/w) * (v_c - v_uc)
            w4 = rearrange(w_cond, "b -> b 1 1 1")
            target = (x + (1.0 - 1.0 / w4) * (x_c - x_uc)).detach()
        else:
            c_cond, w_cond = c, None
            target = x.detach()

        # ---- forward 1: direct prediction (student) + split ratio ----------
        if self.alpha_mode == 'predicted':
            x_hat_p, alpha = model(z_t, t, r, y=c_cond, w=w_cond,
                                   use_flash_attention=self.use_flash_attention)
            alpha_det = alpha.detach()
            alpha_used = alpha if not self.alpha_stopgrad else alpha_det
            s = alpha_used * r + (1 - alpha_used) * t
        else:
            x_hat_p = model(z_t, t, r, y=c_cond, w=w_cond, return_alpha=False,
                            use_flash_attention=self.use_flash_attention)
            alpha = None
            # random split: every s in (r, t) is an equally valid interval, so a
            # learned head / curriculum has no signal source here (M4)
            alpha_det = self.split_min + (1.0 - 2.0 * self.split_min) * torch.rand(
                batch_size, device=device)
            s = alpha_det * r + (1.0 - alpha_det) * t
        alpha4 = rearrange(alpha_det, "b -> b 1 1 1")
        s4 = rearrange(s, "b -> b 1 1 1")

        # ---- forward 2: first leg -----------------------------------------
        x_hat_s = model(z_t, t, s, y=c_cond, w=w_cond, return_alpha=False,
                        use_flash_attention=self.use_flash_attention)
        t_safe = t4.clamp_min(1e-6)
        z_s_hat = (1 - s4 / t_safe) * x_hat_s + (s4 / t_safe) * z_t
        s_hat = z_s_hat                                     # state at time s

        # ---- forward 3: second leg (the composition / teacher) ------------
        z_s_in = s_hat.detach() if self.detach_xs else s_hat
        x_hat_r = model(z_s_in, s, r, y=c_cond, w=w_cond, return_alpha=False,
                        use_flash_attention=self.use_flash_attention)

        zero = torch.zeros((), device=device, dtype=x_hat_p.dtype)

        # ---- 1) direct pixel anchoring (v1 only; v2 sets pixel_weight=0) ---
        l_pixel = F.mse_loss(x_hat_p, target) if self.pixel_weight > 0 else zero
        # ---- 2) composed-path anchoring (the data-side objective) ---------
        l_anchor = F.mse_loss(x_hat_r, target) if self.anchor_weight > 0 else zero
        # ---- 3) consistency distillation: direct <- composed (v2 core, M1) -
        l_cons = (F.mse_loss(x_hat_p, x_hat_r.detach())
                  if self.cons_weight > 0 else zero)
        # ---- 4) retired v1 terms (kept for ablations) ----------------------
        l_freq = (self.freq_consistency(x_hat_p, x_hat_r)
                  if self.freq_weight > 0 else zero)
        l_mid = zero
        if self.mid_weight > 0:
            z_s_true = (z_t + alpha4 * (r4 - t4) * v).detach()
            l_mid = F.mse_loss(s_hat, z_s_true)
        alpha_target = (self._alpha_target(step, total_steps)
                        if self.alpha_mode == 'predicted' else None)
        l_alpha = zero
        if (self.alpha_mode == 'predicted' and self.alpha_weight > 0
                and alpha_target is not None):
            l_alpha = F.mse_loss(alpha, torch.full_like(alpha, float(alpha_target)))

        total = (
            self.pixel_weight * l_pixel
            + self.anchor_weight * l_anchor
            + self.cons_weight * l_cons
            + self.freq_weight * l_freq
            + self.mid_weight * l_mid
            + self.alpha_weight * l_alpha
        )

        alpha_pred = alpha.detach().mean() if alpha is not None else alpha_det.mean()
        stats = {
            'pixel': l_pixel.detach(),
            'anchor': l_anchor.detach(),
            'cons': l_cons.detach(),
            'freq': l_freq.detach(),
            'mid': l_mid.detach(),
            'alpha': l_alpha.detach(),
            'alpha_pred': alpha_pred,
            'alpha_target': torch.tensor(
                float(alpha_target) if alpha_target is not None else -1.0),
        }
        return total, stats

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=1, device='cuda', adaptive=False):
        """
        x-prediction sampler.

        1 step: x_hat = model(z, 1, 0, w) is the final image directly.
        K steps: z_r = (1 - r/t) * x_hat + (r/t) * z_t  (closed-form state update).
        adaptive=True additionally splits every interval; it carries information
        only when the alpha head was trained (alpha_mode='predicted'), otherwise
        a fixed 0.5 split is used.

        w is a model input (CFG distillation), so guidance costs no extra NFE.
        """
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        n = c.shape[0]
        z = torch.randn(n, self.channels, self.image_size, self.image_size, device=device)
        w = self._sampling_w(n, device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((n,), t_vals[i].item(), device=device)
            r = torch.full((n,), t_vals[i + 1].item(), device=device)
            if adaptive:
                if self.alpha_mode == 'predicted':
                    _, alpha = model(z, t, r, y=c, w=w,
                                     use_flash_attention=self.use_flash_attention)
                else:
                    alpha = torch.full((n,), 0.5, device=device)
                s = alpha * r + (1 - alpha) * t
                x_hat_s = model(z, t, s, y=c, w=w, return_alpha=False,
                                use_flash_attention=self.use_flash_attention)
                z = self._state_from_clean(x_hat_s, z, t, s)
                x_hat_r = model(z, s, r, y=c, w=w, return_alpha=False,
                                use_flash_attention=self.use_flash_attention)
                z = self._state_from_clean(x_hat_r, z, s, r)
            else:
                x_hat = model(z, t, r, y=c, w=w, return_alpha=False,
                              use_flash_attention=self.use_flash_attention)
                z = self._state_from_clean(x_hat, z, t, r)

        return self.normer.unnorm(z)
