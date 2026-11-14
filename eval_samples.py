"""Evaluate one-step / two-step / adaptive sampling of a trained AF-PSM model.

The core research question: does the 2-step (split) composition actually improve
over the 1-step direct prediction, and does the adaptive alpha split help?

What it does:
  * samples the same noise seeds with several sampling modes
  * reports sample mean/std and pairwise diversity
  * reports the consistency between modes (mean |x_1step - x_2step| for the same z):
      - large  -> the split path really differs from the direct path
      - ~0     -> the split is a no-op (alpha curriculum collapsed to one step)
  * saves image grids (one PNG per mode) for visual comparison
  * optionally compares sample statistics with the real dataset

Examples:
    python eval_samples.py --config configs/afpsm_mnist.py \
        --ckpt logs/afpsm_mnist_10k_*/ckpts/step_0010000.pt --n_per_class 4

    python eval_samples.py --modes 1,2,2a --outdir eval_10k
"""
import argparse
import importlib.util
import inspect
import os

import torch
from torchvision.utils import make_grid, save_image

from meanflow import AFPSM
from models.dit import MFDiT


def load_config(path):
    spec = importlib.util.spec_from_file_location('eval_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def build_afpsm(cfg):
    mf_cfg = dict(cfg['meanflow'])
    mf_cfg.pop('mode', None)
    mf_cfg['channels'] = cfg['model']['in_channels']
    mf_cfg['image_size'] = cfg['model']['input_size']
    mf_cfg['num_classes'] = cfg['model']['num_classes']
    valid = inspect.signature(AFPSM).parameters
    return AFPSM(**{k: v for k, v in mf_cfg.items() if k in valid})


def parse_mode(s):
    """'1' -> (1, False), '2' -> (2, False), '2a'/'adaptive' -> (2, True)"""
    s = str(s).strip().lower()
    if s == 'adaptive':
        return 2, True
    if s.endswith('a'):
        return int(s[:-1]), True
    return int(s), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/afpsm_mnist.py')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_per_class', type=int, default=4)
    parser.add_argument('--modes', type=str, default='1,2,2a')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--data_stats', type=int, default=0,
                        help='1: also print real-data mean/std for comparison')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = MFDiT(**cfg['model']).to(args.device)
    ckpt = args.ckpt
    if ckpt is None:
        import glob
        cand = sorted(glob.glob('logs/**/ckpts/*.pt', recursive=True))
        if not cand:
            raise FileNotFoundError('no ckpt found; pass --ckpt')
        ckpt = cand[-1]
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()
    print(f'ckpt: {ckpt}')
    print(f'model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params '
          f'| activation={model.output_activation}')

    afpsm = build_afpsm(cfg)
    classes = list(range(cfg['model']['num_classes']))
    modes = [parse_mode(m) for m in args.modes.split(',')]
    outdir = args.outdir or os.path.join('eval', os.path.basename(os.path.dirname(os.path.dirname(ckpt))))
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for steps, adaptive in modes:
        name = f'{"adaptive" if adaptive else "step"}{steps}'
        torch.manual_seed(args.seed)          # same starting noise for every mode
        imgs = afpsm.sample_each_class(
            model, n_per_class=args.n_per_class, classes=classes,
            sample_steps=steps, device=args.device, adaptive=adaptive,
        )
        results[name] = imgs
        grid = make_grid(imgs, nrow=len(classes))
        path = os.path.join(outdir, f'{name}.png')
        save_image(grid, path)
        print(f'[{name}] saved {path} | mean={imgs.mean():.4f} std={imgs.std():.4f}')

    # ---- diversity + cross-mode consistency -------------------------------
    print('\n-- pairwise diversity (mean abs diff between different samples) --')
    for name, imgs in results.items():
        if imgs.shape[0] < 2:
            continue
        d = (imgs[:, None] - imgs[None, :]).abs().mean(dim=(2, 3, 4))
        n = imgs.shape[0]
        off = d.sum() / (n * (n - 1))
        print(f'  {name}: {off:.4f}')

    if 'step1' in results:
        print('\n-- consistency vs one-step (same noise seed; larger = split matters) --')
        for name, imgs in results.items():
            if name == 'step1':
                continue
            d = (results['step1'] - imgs).abs().mean().item()
            print(f'  |step1 - {name}| = {d:.4f}')

    # ---- optional: real-data statistics -----------------------------------
    if args.data_stats:
        try:
            from torch.utils.data import DataLoader
            from data import build_dataset
            ds = build_dataset(cfg)
            loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2)
            batch = next(iter(loader))[0]
            batch = afpsm.normer.norm(batch).to(args.device)
            print(f'\n-- real data stats ({batch.shape[0]} images) --')
            print(f'  normalized: mean={batch.mean():.4f} std={batch.std():.4f}')
            if cfg['meanflow']['normalizer'][0] == 'minmax':
                # samples are reported in [0, 1] (after unnorm), so map data too
                batch01 = (batch + 1) / 2
                print(f'  in [0, 1] : mean={batch01.mean():.4f} std={batch01.std():.4f}')
        except Exception as e:  # noqa: BLE001 - data is optional for this eval
            print(f'\n[data_stats] skipped: {e}')

    print(f'\nall grids written to: {os.path.abspath(outdir)}')


if __name__ == '__main__':
    main()
