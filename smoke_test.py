"""Smoke test for AF-PSM (CPU only, no dataset needed).

Checks:
  1) MFDiT(output_mode='x'): shapes, tanh bound, alpha in [alpha_min, 1]
  2) AFPSM.loss: finite total + 4 sub-losses, backward, all params get grads
  3) fixed-batch overfit: loss decreases
  4) frequency loss variants ('rel_hp', 'mse'): scalar, finite
  5) x-pred sampler: 1-step / 2-step / 2-step adaptive
  6) baseline regression: MFDiT(output_mode='uv') + JVP MeanFlow.loss still runs

Run:  .venv\\Scripts\\python.exe smoke_test.py   (cwd = MeanFlow)
"""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch

from meanflow import MeanFlow, AFPSM
from models.dit import MFDiT


def build_model(C=1, H=8, mode='x', alpha_min=0.01):
    return MFDiT(
        input_size=H,
        patch_size=2,
        in_channels=C,
        dim=64,
        depth=2,
        num_heads=4,
        num_classes=10,
        max_cfg=7.5,
        output_mode=mode,
        output_activation='tanh',
        alpha_min=alpha_min,
    )


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cpu'
    B, C, H = 4, 1, 8
    x = torch.rand(B, C, H, H)                 # images in [0, 1]
    c = torch.randint(0, 10, (B,))
    t = torch.rand(B) * 0.9 + 0.1              # 0.1 .. 1.0
    r = torch.rand(B) * 0.5                    # < t

    # ---------------- 1) forward -------------------------------------------
    model = build_model(C, H)
    model.eval()
    z = torch.randn(B, C, H, H)
    x_hat, alpha = model(z, t, r, y=c)
    assert x_hat.shape == (B, C, H, H), x_hat.shape
    assert torch.all(x_hat.abs() <= 1.0)
    assert alpha.shape == (B,)
    assert (alpha >= 0.01 - 1e-6).all() and (alpha <= 1.0).all()
    assert model(z, t, r, y=c, return_alpha=False).shape == (B, C, H, H)
    print(f"[1] forward ok | x_hat {tuple(x_hat.shape)} | alpha "
          f"{alpha.detach().numpy().round(3)}")

    # ---------------- 2) loss + backward -----------------------------------
    afpsm = AFPSM(
        channels=C,
        image_size=H,
        num_classes=10,
        normalizer=['minmax', None, None],
        time_dist=['uniform'],
        alpha_curriculum='linear',
        alpha_weight=0.1,
        freq_weight=0.5,
        mid_weight=0.5,
        alpha_stopgrad=True,
        freq_loss='rel_hp',
        freq_gamma=1.0,
    )
    model.train()
    total, stats = afpsm.loss(model, x, c, step=0, total_steps=1000)
    assert torch.isfinite(total), total
    for k in ('pixel', 'freq', 'mid', 'alpha', 'alpha_pred', 'alpha_target'):
        assert k in stats, k
        assert torch.isfinite(stats[k]).all(), k
    assert abs(stats['alpha_target'].item() - 1.0) < 1e-6
    total.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_zero = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() == 0
    )
    assert n_none == 0, f"{n_none} params without grad"
    print(f"[2] loss ok | total {total.item():.4f} | "
          + " ".join(f"{k}={stats[k].item():.4f}"
                     for k in ('pixel', 'freq', 'mid', 'alpha'))
          + f" | alpha_pred={stats['alpha_pred'].item():.3f} | "
          f"zero-grad params={n_zero} (expected >0 at init: DiT zero-inits adaLN gating)")

    # ---------------- 3) fixed-batch overfit -------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(40):
        # same t, r and noise every step -> a deterministic fixed objective
        torch.manual_seed(1234)
        np.random.seed(1234)
        opt.zero_grad()
        total, _ = afpsm.loss(model, x, c, step=0, total_steps=1000)
        total.backward()
        opt.step()
        if step == 0:
            first = total.item()
        last = total.item()
    assert last < first, (first, last)
    print(f"[3] overfit ok | loss {first:.4f} -> {last:.4f} (40 steps)")

    # gradient must now flow through every parameter, including the 3-forward chain
    opt.zero_grad()
    torch.manual_seed(7)
    np.random.seed(7)
    total, _ = afpsm.loss(model, x, c, step=0, total_steps=1000)
    total.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert len(no_grad) == 0, f"params with no gradient after warmup: {no_grad}"
    print(f"[3b] gradient flow ok | all {sum(1 for _ in model.parameters())} "
          f"tensors receive non-zero gradients after warmup")

    # ---------------- 4) frequency loss variants ---------------------------
    a = torch.rand(B, C, H, H) * 2 - 1
    b = a + 0.05 * torch.randn_like(a)
    for kind in ('rel_hp', 'mse'):
        afpsm.freq_loss = kind
        lf = afpsm.freq_consistency(a, b)
        assert lf.ndim == 0 and torch.isfinite(lf), (kind, lf)
    afpsm.freq_loss = 'rel_hp'
    print("[4] freq loss ok | rel_hp + mse are finite scalars")

    # ---------------- 5) samplers ------------------------------------------
    model.eval()
    classes = [0, 1, 2, 3]
    for steps, adaptive in ((1, False), (2, False), (2, True)):
        imgs = afpsm.sample_each_class(
            model, n_per_class=1, classes=classes,
            sample_steps=steps, device=device, adaptive=adaptive,
        )
        assert imgs.shape == (4, C, H, H), imgs.shape
        assert torch.isfinite(imgs).all()
        assert imgs.min() >= -1e-6 and imgs.max() <= 1 + 1e-6
        print(f"    sampler ok | steps={steps} adaptive={adaptive} "
              f"range=[{imgs.min():.3f}, {imgs.max():.3f}]")
    print("[5] sampler ok | 1-step / 2-step / 2-step adaptive")

    # ---------------- 6) baseline regression (JVP path) ---------------------
    model_uv = build_model(C, H, mode='uv')
    mf = MeanFlow(
        channels=C, image_size=H, num_classes=10,
        normalizer=['minmax', None, None], time_dist=['uniform'],
    )
    loss_uv, stats_uv = mf.loss(model_uv, x, c)
    assert torch.isfinite(loss_uv)
    print(f"[6] baseline ok | JVP MeanFlow loss {loss_uv.item():.4f} | "
          f"keys={sorted(stats_uv)}")

    # ---------------- 7) real config wiring (same path as train.py) --------
    cfg_path = Path(__file__).resolve().parent / 'configs' / 'afpsm_mnist_10k.py'
    spec = importlib.util.spec_from_file_location('cfg_afpsm_mnist_10k', cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = cfg_mod.config
    model_real = MFDiT(**cfg['model'])
    mf_cfg = dict(cfg['meanflow'])
    mf_mode = mf_cfg.pop('mode')
    mf_cfg['channels'] = cfg['model']['in_channels']
    mf_cfg['image_size'] = cfg['model']['input_size']
    mf_cfg['num_classes'] = cfg['model']['num_classes']
    valid = inspect.signature(AFPSM).parameters
    afpsm_real = AFPSM(**{k: v for k, v in mf_cfg.items() if k in valid})
    x_real = torch.rand(2, 1, 32, 32)
    c_real = torch.randint(0, 10, (2,))
    loss_real, stats_real = afpsm_real.loss(
        model_real, x_real, c_real, step=100, total_steps=cfg['n_steps']
    )
    loss_real.backward()
    assert torch.isfinite(loss_real)
    n_params = sum(p.numel() for p in model_real.parameters())
    print(f"[7] real config wiring ok | mode={mf_mode} | "
          f"{n_params / 1e6:.2f}M params | loss={loss_real.item():.4f} | "
          f"alpha={stats_real['alpha_pred'].item():.3f} | "
          f"grad_ckpt={cfg['model'].get('grad_checkpoint', False)} | "
          f"flash={cfg['meanflow'].get('use_flash_attention', False)}")

    # ---------------- 8) memory-saving switches (grad ckpt / SDPA flag) -----
    model_ckpt = build_model(C, H)
    model_ckpt.grad_checkpoint = True          # exercise the checkpoint branch
    model_ckpt.train()
    total_ck, _ = afpsm.loss(model_ckpt, x, c, step=0, total_steps=1000)
    total_ck.backward()
    assert torch.isfinite(total_ck)
    n_ck_none = sum(1 for p in model_ckpt.parameters()
                    if p.requires_grad and p.grad is None)
    assert n_ck_none == 0
    print(f"[8] grad_checkpoint ok | loss {total_ck.item():.4f} | all params get grads")

    afpsm_sdpa = AFPSM(
        channels=C, image_size=H, num_classes=10,
        normalizer=['minmax', None, None], time_dist=['uniform'],
        use_flash_attention=True,          # on CPU this falls back to native attention
    )
    model_sdpa = build_model(C, H)
    total_sdpa, _ = afpsm_sdpa.loss(model_sdpa, x, c, step=0, total_steps=1000)
    total_sdpa.backward()
    assert torch.isfinite(total_sdpa)
    imgs_sdpa = afpsm_sdpa.sample_each_class(
        model_sdpa, n_per_class=1, classes=[0, 1], sample_steps=2, device='cpu'
    )
    assert torch.isfinite(imgs_sdpa).all()
    print(f"[8b] use_flash_attention plumbing ok | loss {total_sdpa.item():.4f} | "
          f"sampler ok")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
