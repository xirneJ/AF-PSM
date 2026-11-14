import argparse
import importlib.util
import inspect
import math
import os
import shutil
import time
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import build_dataset, cycle, resolve_data_root
from meanflow import MeanFlow, AFPSM
from models.dit import MFDiT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_suffix', type=str, default=None)
    # quick resource/scale overrides, so a shared GPU can be probed without
    # editing the config file
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--flash', type=int, default=None, choices=[0, 1],
                        help='1: memory-efficient SDPA attention (AF-PSM only)')
    parser.add_argument('--ckpt', type=int, default=None, choices=[0, 1],
                        help='1: gradient checkpointing in the DiT blocks')
    parser.add_argument('--sample_steps', type=int, default=None)
    parser.add_argument('--data_root', type=str, default=None,
                        help='dataset root (relative paths resolve against the repo root, '
                             'e.g. data/mnist)')
    parser.add_argument('--output_activation', type=str, default=None,
                        choices=['tanh', 'clamp', 'none'],
                        help='AF-PSM clean-image head activation (diagnostics)')
    parser.add_argument('--precision', type=str, default=None,
                        choices=['bf16', 'fp16', 'no'],
                        help='mixed precision (diagnostics)')
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('train_config', args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.config

    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.n_steps is not None:
        cfg['n_steps'] = args.n_steps
    if args.sample_steps is not None:
        cfg['sample_steps'] = args.sample_steps
    if args.flash is not None:
        cfg['meanflow']['use_flash_attention'] = bool(args.flash)
    if args.ckpt is not None:
        cfg['model']['grad_checkpoint'] = bool(args.ckpt)
    if args.data_root is not None:
        cfg['data_root'] = args.data_root
    if args.output_activation is not None:
        cfg['model']['output_activation'] = args.output_activation
    if args.precision is not None:
        cfg['mixed_precision'] = args.precision
    if args.lr is not None:
        cfg['lr'] = args.lr

    accelerator = Accelerator(
        mixed_precision=cfg.get('mixed_precision', 'bf16'),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.is_main_process:
        print(f"[override] batch_size={cfg['batch_size']} n_steps={cfg['n_steps']} "
              f"flash={cfg['meanflow'].get('use_flash_attention', False)} "
              f"grad_checkpoint={cfg['model'].get('grad_checkpoint', False)} "
              f"output_activation={cfg['model'].get('output_activation', 'tanh')} "
              f"precision={cfg.get('mixed_precision', 'bf16')}")
        print(f"[data] dataset={cfg['dataset']} root={resolve_data_root(cfg)}")

    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(cfg),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=cfg['num_workers'],
    )

    use_latent = cfg.get('latent', False)
    vae = None
    if use_latent:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(cfg['vae_id']).to(accelerator.device)

    model = MFDiT(**cfg['model']).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    scheduler = None
    sched_cfg = cfg.get('scheduler')
    if sched_cfg is not None:
        warmup_steps = sched_cfg.get('warmup_steps', 0)
        min_lr_ratio = sched_cfg.get('min_lr_ratio', 0.1)
        total_steps = cfg['n_steps']

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(step, 1) / warmup_steps
            if step >= total_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    mf_cfg = dict(cfg['meanflow'])
    mf_mode = mf_cfg.pop('mode', 'i-meanflow')
    mf_cfg['channels'] = cfg['model']['in_channels']
    mf_cfg['image_size'] = cfg['model']['input_size']
    mf_cfg['num_classes'] = cfg['model']['num_classes']
    is_afpsm = mf_mode == 'af-psm'
    if is_afpsm:
        # keep only the kwargs AFPSM actually understands
        valid = inspect.signature(AFPSM).parameters
        meanflow = AFPSM(**{k: v for k, v in mf_cfg.items() if k in valid})
    else:
        meanflow = MeanFlow(mode=mf_mode, **mf_cfg)

    if vae is not None:
        model, vae, optimizer, train_dataloader = accelerator.prepare(
            model, vae, optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        run_name = (
            f"{cfg['name']}_{args.run_suffix}"
            if args.run_suffix else
            f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        cfg.update({
            'run_name': run_name,
            'log_dir': os.path.join('logs', run_name),
            'image_dir': os.path.join('logs', run_name, 'images'),
            'ckpt_dir': os.path.join('logs', run_name, 'ckpts'),
        })
        os.makedirs(cfg['log_dir'], exist_ok=True)
        os.makedirs(cfg['image_dir'], exist_ok=True)
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        cfg['log_file'] = os.path.join(cfg['log_dir'], 'train.log')
        shutil.copy2(args.config, os.path.join(cfg['log_dir'], 'config.py'))
        with open(cfg['log_file'], 'w') as f:
            f.write(f"run_name: {run_name}\nconfig: {args.config}\n\n")
    accelerator.wait_for_everyone()

    model_module = model.module if hasattr(model, 'module') else model
    train_dataloader = cycle(train_dataloader)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model_module.parameters())
        print(f"MF mode: {mf_mode}")
        print(f"Model size: {n_params / 1e6:.2f}M")
        with open(cfg['log_file'], 'a') as f:
            f.write(f"MF mode: {mf_mode}\n")
            f.write(f"Model size: {n_params / 1e6:.2f}M\n\n")

    global_step = 0
    grad_clip = cfg['grad_clip']
    losses = log_grad_norm = 0.0
    stats_sum = {}
    log_t0 = time.time()

    model.train()
    pbar = tqdm(
        range(cfg['n_steps']),
        desc=cfg['name'],
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )
    for _ in pbar:
        x, c = next(train_dataloader)
        x = x.to(accelerator.device)
        c = c.to(accelerator.device)
        if use_latent:
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample()

        if is_afpsm:
            loss, mse_val = meanflow.loss(
                model, x, c, step=global_step, total_steps=cfg['n_steps']
            )
        else:
            loss, mse_val = meanflow.loss(model, x, c)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        losses += loss.item()
        for k, v in mse_val.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v.item()
        log_grad_norm += grad_norm.item()

        if accelerator.is_main_process:
            post = dict(Loss=f"{loss.item():.4f}")
            for k in ('pixel', 'freq', 'mid'):
                if k in mse_val:
                    post[k.capitalize()] = f"{mse_val[k].item():.4f}"
            if 'alpha_pred' in mse_val:
                post['A'] = f"{mse_val['alpha_pred'].item():.3f}"
            if 'mf_v_mse' in mse_val:
                post['MF_V_MSE'] = f"{mse_val['mf_v_mse'].item():.4f}"
            pbar.set_postfix(**post, refresh=False)

        if accelerator.is_main_process and global_step % cfg['log_step'] == 0:
            s = cfg['log_step']
            elapsed = time.time() - log_t0
            mem_peak = (torch.cuda.max_memory_allocated() / 2**30
                        if torch.cuda.is_available() else 0.0)
            with open(cfg['log_file'], 'a') as f:
                lr = optimizer.param_groups[0]['lr']
                parts = "    ".join(f"{k}: {v / s:.6f}" for k, v in stats_sum.items())
                f.write(
                    f"{time.asctime(time.localtime(time.time()))}\n"
                    f"Step: {global_step}    LR: {lr:.6f}    Grad_Norm: {log_grad_norm / s:.6f}    "
                    f"Step_time: {elapsed / s:.3f}s    Mem_peak: {mem_peak:.2f}GB\n"
                    f"Loss: {losses / s:.6f}    {parts}\n"
                )
            losses = log_grad_norm = 0.0
            stats_sum = {}
            log_t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        if global_step % cfg['sample_step'] == 0:
            if accelerator.is_main_process:
                if is_afpsm:
                    z = meanflow.sample_each_class(
                        model_module,
                        n_per_class=1,
                        classes=cfg['sample_classes'],
                        sample_steps=cfg.get('sample_steps', 1),
                        device=accelerator.device,
                        adaptive=cfg.get('sample_adaptive', False),
                    )
                else:
                    z = meanflow.sample_each_class(
                        model_module, n_per_class=1, classes=cfg['sample_classes']
                    )
                if use_latent:
                    with torch.no_grad():
                        z = vae.decode(z).sample
                    z = z * 0.5 + 0.5
                save_image(
                    make_grid(z, nrow=cfg['sample_nrow'] or int(z.shape[0] ** 0.5)),
                    os.path.join(cfg['image_dir'], f'step_{global_step:07d}.png'),
                )
            accelerator.wait_for_everyone()
            model.train()

    if accelerator.is_main_process:
        accelerator.save(
            model_module.state_dict(),
            os.path.join(cfg['ckpt_dir'], f'step_{global_step:07d}.pt'),
        )
