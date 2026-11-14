"""FID evaluation for AF-PSM and MeanFlow baselines.

Fair-comparison protocol
------------------------
  * same detector: torchvision InceptionV3 (ImageNet weights), 2048-d pool3 features
  * same reference: dataset *test* split, images in [0, 1]
  * same number of class-balanced generated samples (default 10,000)
  * report one FID per sampling mode (1-step / 2-step / adaptive)

Caveat: FID on 32x32 datasets (MNIST / CIFAR) goes through a resize to 299 and is
NOT comparable to literature FID numbers. Use it only for the *relative* comparison
between your own models (same detector, same reference set, same sample count).

Examples
--------
    # AF-PSM (checkpoint from a 10k-step run)
    python fid_eval.py --config configs/afpsm_mnist_10k.py \
        --ckpt logs/afpsm_mnist_10k_*/ckpts/step_0010000.pt \
        --modes 1,2,2a --n_samples 10000 --name afpsm_10k --json fid_results.jsonl

    # MeanFlow / iMF baseline (trained with configs/mnist.py)
    python fid_eval.py --config configs/mnist.py \
        --ckpt logs/<baseline_run>/ckpts/step_0010000.pt \
        --modes 1,2 --n_samples 10000 --name meanflow_10k --json fid_results.jsonl

    # inspect the collected results
    python - <<'PY'
    import json
    for line in open('fid_results.jsonl'):
        r = json.loads(line)
        print(f"{r['name']:>28}  mode={r['mode']:>4}  n={r['n']:>6}  FID={r['fid']:8.2f}")
    PY
"""
import argparse
import glob
import importlib.util
import inspect
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

from data import build_eval_dataset
from meanflow import AFPSM, MeanFlow
from models.dit import MFDiT

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(path):
    spec = importlib.util.spec_from_file_location('fid_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def parse_mode(s):
    """'1' -> (1, False), '2' -> (2, False), '2a'/'adaptive' -> (2, True)"""
    s = str(s).strip().lower()
    if s == 'adaptive':
        return 2, True
    if s.endswith('a'):
        return int(s[:-1]), True
    return int(s), False


class InceptionFeatures(torch.nn.Module):
    """Standard FID detector: 2048-d pool3 features of InceptionV3."""

    def __init__(self):
        super().__init__()
        self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.net.eval()
        self._out = None
        self.net.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._out = output.flatten(1)

    @torch.no_grad()
    def forward(self, x):
        self.net(x)
        return self._out


@torch.no_grad()
def features(imgs, detector, device, batch=64):
    out = []
    for i in range(0, len(imgs), batch):
        x = imgs[i:i + batch].to(device).float()
        if x.shape[1] == 1:                     # grayscale -> RGB
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] > 3:
            x = x[:, :3]
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        out.append(detector(x).cpu())
    return torch.cat(out)


@torch.no_grad()
def reference_images(cfg, n, device, num_workers=4):
    ds = build_eval_dataset(cfg, train=False)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=num_workers,
                        pin_memory=(device.type != 'cpu'))
    chunks, total = [], 0
    for batch in loader:
        x = batch[0].float()
        chunks.append(x)
        total += len(x)
        if total >= n:
            break
    return torch.cat(chunks)[:n]


def fid_from_features(f1, f2):
    """FID = |mu1-mu2|^2 + tr(S1+S2-2*sqrt(S1 S2)), with a PSD-safe trace term."""
    f1 = f1.double().numpy()
    f2 = f2.double().numpy()
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    sv = np.linalg.svd(s1 @ s2, compute_uv=False)
    covmean_trace = np.sqrt(np.clip(sv, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2)
                 - 2.0 * covmean_trace)


def generate_samples(sampler, model, n_samples, n_classes, mode, device):
    steps, adaptive = parse_mode(mode)
    n_per_class = max(1, n_samples // n_classes)
    kwargs = dict(n_per_class=n_per_class, classes=list(range(n_classes)),
                  sample_steps=steps, device=device)
    if isinstance(sampler, AFPSM):
        kwargs['adaptive'] = adaptive
    return sampler.sample_each_class(model, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--modes', type=str, default='1',
                        help="comma list, e.g. 1,2,2a ('a' = adaptive alpha split)")
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--n_ref', type=int, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader workers for the reference split')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', type=str, default=None,
                        help='append one JSON line per (model, mode) result')
    parser.add_argument('--cache_dir', type=str, default='.cache/fid')
    parser.add_argument('--no_cache', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = args.ckpt
    if ckpt is None:
        cand = sorted(glob.glob('logs/**/ckpts/*.pt', recursive=True))
        if not cand:
            raise FileNotFoundError('no checkpoint found, pass --ckpt')
        ckpt = cand[-1]
    name = args.name or os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    device = torch.device(args.device)

    model = MFDiT(**cfg['model']).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    mf_cfg = dict(cfg['meanflow'])
    mf_mode = mf_cfg.pop('mode', 'i-meanflow')
    mf_cfg['channels'] = cfg['model']['in_channels']
    mf_cfg['image_size'] = cfg['model']['input_size']
    mf_cfg['num_classes'] = cfg['model']['num_classes']
    if mf_mode == 'af-psm':
        valid = inspect.signature(AFPSM).parameters
        sampler = AFPSM(**{k: v for k, v in mf_cfg.items() if k in valid})
    else:
        sampler = MeanFlow(mode=mf_mode, **mf_cfg)

    print(f'[{name}] ckpt: {ckpt}')
    print(f'[{name}] sampler={type(sampler).__name__} mode={mf_mode} '
          f'activation={getattr(model, "output_activation", "-")}')

    detector = InceptionFeatures().to(device).eval()

    n_ref = args.n_ref or args.n_samples
    cache = os.path.join(args.cache_dir,
                         f'ref_{cfg["dataset"]}_{cfg["model"]["input_size"]}_{n_ref}.npz')
    if not args.no_cache and os.path.exists(cache):
        ref_feats = torch.from_numpy(np.load(cache)['f'])
        print(f'[{name}] reference features cached: {cache} (n={len(ref_feats)})')
    else:
        ref_imgs = reference_images(cfg, n_ref, device, num_workers=args.num_workers)
        print(f'[{name}] reference: {len(ref_imgs)} images from '
              f'{cfg["dataset"]} test split')
        ref_feats = features(ref_imgs, detector, device, args.batch)
        if not args.no_cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            np.savez(cache, f=ref_feats.numpy())
            print(f'[{name}] cached reference features -> {cache}')

    results = []
    n_classes = cfg['model']['num_classes']
    for mode in args.modes.split(','):
        torch.manual_seed(args.seed)
        imgs = generate_samples(sampler, model, args.n_samples, n_classes, mode, device)
        feats = features(imgs, detector, device, args.batch)
        score = fid_from_features(feats, ref_feats)
        print(f'[{name}] mode={mode.strip():>4}  n={len(imgs)}  FID={score:.2f}')
        results.append(dict(name=name, config=args.config, ckpt=ckpt,
                            sampler=type(sampler).__name__, mode=mode.strip(),
                            n=len(imgs), n_ref=len(ref_feats), fid=score))

    if args.json:
        with open(args.json, 'a') as f:
            for row in results:
                f.write(json.dumps(row) + '\n')
        print(f'[{name}] appended {len(results)} rows -> {args.json}')


if __name__ == '__main__':
    main()
