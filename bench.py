"""Measure AF-PSM step time and peak GPU memory for one (batch, flash, ckpt) config.

Run ONE configuration per process: a fresh CUDA allocator is the whole point
(the inline multi-config benchmark can OOM because of allocator carry-over).

    python bench.py --batch 16 --flash 1 --ckpt 1

Sweep from the shell (each line is a separate process):

    for b in 8 16 32 64; do
      for f in 0 1; do
        for c in 0 1; do
          python bench.py --batch $b --flash $f --ckpt $c
        done
      done
    done

Read it as: pick the largest batch whose peak memory is comfortably below the
GPU's free memory (leave ~1.5-2 GB headroom for the allocator).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import time

import torch

from meanflow import AFPSM
from models.dit import MFDiT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--flash', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ckpt', type=int, default=1, choices=[0, 1])
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('CUDA not available')
        return 0
    dev = 'cuda'
    free, total = torch.cuda.mem_get_info()

    model = MFDiT(
        input_size=32, patch_size=2, in_channels=1,
        dim=args.dim, depth=args.depth, num_heads=8, num_classes=10, max_cfg=7.5,
        output_mode='x', output_activation='tanh', alpha_min=0.01,
        grad_checkpoint=bool(args.ckpt),
    ).to(dev)
    lf = AFPSM(
        channels=1, image_size=32, num_classes=10,
        normalizer=['minmax', None, None], time_dist=['lognorm', -0.4, 1.0],
        use_flash_attention=bool(args.flash),
    )
    x = torch.rand(args.batch, 1, 32, 32, device=dev)      # images in [0, 1]
    c = torch.randint(0, 10, (args.batch,), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one_step():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = lf.loss(model, x, c, step=0, total_steps=10000)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss

    tag = f'batch={args.batch:3d} flash={args.flash} ckpt={args.ckpt}'
    try:
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            loss = one_step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.steps
        peak = torch.cuda.max_memory_allocated() / 2**30
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'{tag} | {dt * 1000:7.0f} ms/step | peak {peak:5.2f} GB | '
              f'gpu_free_before {free / 2**30:5.2f} GB | '
              f'{n_params:.2f}M params | loss {loss.item():.4f}')
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if 'out of memory' in str(e).lower():
            print(f'{tag} | OOM')
        else:
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
