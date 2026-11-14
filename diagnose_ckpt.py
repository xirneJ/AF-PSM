"""Inspect a saved AF-PSM checkpoint for mode collapse / tanh saturation.

Examples:
    python diagnose_ckpt.py                       # latest logs/afpsm_*/ckpts/*.pt
    python diagnose_ckpt.py --ckpt logs/.../ckpts/step_00010000.pt

What it reports (and how to read it):
  * output mean/std            -> a constant image has std ~ 0
  * input sensitivity          -> |f(z1)-f(z2)| for two random inputs; ~0 means the
                                  network ignores its input (collapse)
  * pre-tanh |z| stats         -> large |z| + a high frac(|z|>3) means the tanh
                                  head is saturated (gradients die there)
  * per-sample spatial std     -> std within one image; ~0 means a uniform image
  * alpha                      -> what split ratio the alpha head predicts
"""
import argparse
import glob
import importlib.util
import os

import torch

from models.dit import MFDiT


def load_config(path):
    spec = importlib.util.spec_from_file_location('diag_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--config', type=str, default='configs/afpsm_mnist_10k.py')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--output_activation', type=str, default=None,
                        choices=['tanh', 'clamp', 'none'],
                        help='override the config activation (use tanh to diagnose '
                             'checkpoints trained with the old tanh head)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = MFDiT(**cfg['model']).to(args.device)
    ckpt = args.ckpt
    if ckpt is None:
        candidates = sorted(glob.glob('logs/**/ckpts/*.pt', recursive=True))
        if not candidates:
            raise FileNotFoundError('no checkpoint under logs/**/ckpts/*.pt')
        ckpt = candidates[-1]
    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state)
    if args.output_activation is not None:
        model.output_activation = args.output_activation
    model.eval()
    print(f'ckpt: {ckpt}')
    print(f'model: output_mode={model.output_mode} '
          f'activation={model.output_activation} '
          f'params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # capture the pre-activation of the clean-image head
    pre = {}

    def _hook(module, inputs, output):
        pre['z'] = output.detach()

    handle = model.final_layer_x.linear.register_forward_hook(_hook)
    dev, n = args.device, args.n
    torch.manual_seed(0)
    z1 = torch.randn(n, 1, 32, 32, device=dev)
    z2 = torch.randn(n, 1, 32, 32, device=dev)
    c = torch.arange(n, device=dev) % 10
    t = torch.ones(n, device=dev)        # t=1 -> r=0 : the one-step generation path
    r = torch.zeros(n, device=dev)

    with torch.no_grad():
        out1, alpha = model(z1, t, r, y=c)
    z_pre = pre['z']
    with torch.no_grad():
        out2, _ = model(z2, t, r, y=c)
    handle.remove()

    print(f'output: mean={out1.mean().item():.4f} std={out1.std().item():.4f} '
          f'min={out1.min().item():.4f} max={out1.max().item():.4f}')
    print(f'input sensitivity |f(z1)-f(z2)| mean='
          f'{(out1 - out2).abs().mean().item():.6f}  '
          f'(~0 => output ignores the input)')
    zabs = z_pre.abs()
    print(f'pre-tanh |z|: mean={zabs.mean().item():.3f} '
          f'max={zabs.max().item():.3f} '
          f'frac(|z|>3)={(zabs > 3).float().mean().item():.3f} '
          f'frac(|z|>5)={(zabs > 5).float().mean().item():.3f}')
    print(f'output saturation: frac(|out|>0.99)={(out1.abs() > 0.99).float().mean().item():.3f}')
    print(f'per-sample spatial std: '
          f'{[round(float(out1[i].std()), 4) for i in range(n)]}')
    print(f'alpha mean={alpha.mean().item():.4f}')
    print('verdict: ' + (
        'COLLAPSED (output ~ input-independent / near-constant)' if
        (out1 - out2).abs().mean().item() < 1e-3 or out1.std().item() < 1e-2
        else 'output varies with input (no constant collapse detected)'
    ) + ('; tanh SATURATED' if (zabs > 3).float().mean().item() > 0.5 else ''))


if __name__ == '__main__':
    main()
