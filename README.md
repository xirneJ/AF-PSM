<img src="assets/meanflow.gif" width="2000">

# MeanFlow

PyTorch implementation of [Mean Flows for One-step Generative Modeling](https://arxiv.org/pdf/2505.13447) (MeanFlow) and [Improved Mean Flows](https://arxiv.org/abs/2512.02012) (iMF).

> **Note:** Unofficial implementation, based on the papers above and the official JAX repo [imeanflow](https://github.com/Lyy-iiis/imeanflow).

> Contributions and feedback are welcome! Feel free to open an issue or pull request.

## Updates

**2026.06.13**
  - MeanFlow and iMF training (`meanflow.mode`: `"meanflow"` / `"i-meanflow"`)
  - Dual-head DiT: `u` for MeanFlow, `v` for flow matching
  - CFG scale as model input with CFG distillation (`cfg_scale`; `None` to disable)
  - JVP under `no_grad` for `dudt` only; separate grad-enabled forward pass for optimization
  - Config-based training for MNIST, CIFAR-10, and ImageNet latent

**AF-PSM**
  - `meanflow.mode: "af-psm"`: JVP-free interval-splitting training directly in pixel space
  - Single `tanh` clean-image head + `alpha` split-ratio head (global pooling + MLP)
  - Frequency-consistency loss; 3 shared-parameter forwards; 4-loss objective
  - `configs/afpsm_mnist.py`, `configs/afpsm_cifar10.py`; `configs/smoke_cpu.py` + `smoke_test.py` for a CPU sanity check

## AF-PSM (Adaptive Frequency-consistent Pixel Split-MeanFlow)

AF-PSM replaces MeanFlow's JVP-based differential identity with [SplitMeanFlow](https://arxiv.org/abs/2507.16884)'s algebraic interval-splitting identity, re-expressed in **pixel space** (predict the image instead of the average velocity). No `torch.autograd.functional.jvp` is used anywhere, so Flash Attention can be enabled unconditionally.

- trajectory convention: `z_t = (1-t) x + t e`, `v = e - x`, `t > r` while training
- the network predicts the clean image directly: `x_hat, alpha = model(z_t, t, r)` with `alpha` in `[alpha_min, 1]`; the clean-image head is **unbounded by default** (`output_activation='none'` — `tanh` saturates on the minmax ±1 targets and kills gradients; `'tanh'`/`'clamp'` remain for ablation, and samples are always clipped to `[0, 1]` by the normalizer)
- the dynamic split point is `s = alpha * r + (1 - alpha) * t`; the intermediate sample is reconstructed in closed form from the linearity of the path: `z_s = (1 - s/t) * x_hat_s + (s/t) * z_t`
- 3 shared-parameter forwards, 4 losses (pixel anchoring, frequency consistency, exact midpoint supervision, alpha curriculum), zero JVP

```bash
python train.py --config configs/afpsm_mnist.py                  # MNIST (GPU)
python train.py --config configs/afpsm_cifar10.py                # CIFAR-10 (GPU)
python train.py --config configs/afpsm_mnist_10k.py              # short 10k-step validation run
python train.py --config configs/smoke_cpu.py                    # ~200-step CPU sanity check
```

Shared-GPU helpers:

```bash
# measure step time / peak memory for one (batch, flash, ckpt) config;
# run one config per process (fresh CUDA allocator)
python bench.py --batch 16 --flash 1 --ckpt 1

# CLI overrides, no config edit needed
python train.py --config configs/afpsm_mnist_10k.py --batch_size 16 --flash 1 --ckpt 1
```

`train.py` logs `Step_time` and `Mem_peak` every `log_step`, which makes it easy to tune `batch_size` on a shared GPU.

Switches in the `meanflow` block: `alpha_curriculum` (`linear` = 1 -> 0.01, plus `to_half`, `triangular`, `const`, `none`), `freq_loss` (`rel_hp` = ortho rfft2 with soft high-pass weighting and energy normalization, or the literal `mse`), `alpha_weight` / `freq_weight` / `mid_weight`, `alpha_stopgrad`, `detach_xs`, `use_flash_attention` (memory-efficient SDPA; safe now that there is no JVP). `model.grad_checkpoint=True` trades ~20-30% compute for much lower activation memory.

## FID evaluation

`fid_eval.py` computes FID with a standard InceptionV3 detector (torchvision ImageNet weights, auto-downloaded on first use) and appends one JSONL row per (model, sampling mode), so models can be compared in one table.

Fair-comparison protocol: **same detector, same reference set (dataset test split), same number of class-balanced samples** (default 10,000). FID on 32x32 data goes through the Inception resize to 299, so absolute numbers are **not** comparable to literature FID — use them only for the *relative* comparison between your own models.

```bash
# AF-PSM: 1-step / 2-step / adaptive
python fid_eval.py --config configs/afpsm_mnist_10k.py \
  --ckpt logs/<afpsm_run>/ckpts/step_0010000.pt \
  --modes 1,2,2a --n_samples 10000 --name afpsm_10k --json fid_results.jsonl

# MeanFlow / iMF baseline (same budget, CFG disabled)
python fid_eval.py --config configs/mnist.py \
  --ckpt logs/<baseline_run>/ckpts/step_0010000.pt \
  --modes 1,2 --n_samples 10000 --name meanflow_10k --json fid_results.jsonl

# print the collected table
python - <<'PY'
import json
for line in open('fid_results.jsonl'):
    r = json.loads(line)
    print(f"{r['name']:>28}  mode={r['mode']:>4}  n={r['n']:>6}  FID={r['fid']:8.2f}")
PY
```

Train a matched MeanFlow/iMF baseline (same steps, batch, schedule; CFG disabled for a fair comparison with AF-PSM):

```bash
python train.py --config configs/mnist.py --n_steps 10000 --batch_size 32 \
  --warmup_steps 500 --no_cfg --run_suffix imf10k
```

## Usage

`i-meanflow` (`meanflow.mode: "i-meanflow"`) is more stable and recommended for your projects. `meanflow` mode is kept for reference.

```bash
pip install torch accelerate torchvision einops tqdm diffusers
```

Datasets live under `<repo root>/data/` (git-ignored): e.g. `data/mnist/MNIST/...`, `data/cifar/cifar-10-batches-py/...`, `data/imagenet/<class>/...`. Relative `data_root` paths are resolved against the repo root (not the current directory), so training can be launched from anywhere. Override the location with `--data_root` or the `AFPSM_DATA_ROOT` environment variable to move the dataset tree to another disk.

```bash
# single GPU
python train.py --config configs/mnist.py

# custom run name
python train.py --config configs/cifar10.py --run_suffix exp1

# multi-GPU
accelerate launch --num_processes 2 train.py --config configs/mnist.py
```

| Config | Dataset |
|--------|---------|
| `configs/mnist.py` | MNIST |
| `configs/cifar10.py` | CIFAR-10 |
| `configs/imagenet_latent.py` | ImageNet (latent, VAE) |
| `configs/afpsm_mnist.py` | MNIST (AF-PSM, JVP-free) |
| `configs/afpsm_mnist_10k.py` | MNIST (AF-PSM, 10k-step validation run) |
| `configs/afpsm_cifar10.py` | CIFAR-10 (AF-PSM, JVP-free) |
| `configs/smoke_cpu.py` | MNIST, tiny CPU sanity check |

Common config fields: `n_steps`, `batch_size`, `grad_clip`, `mixed_precision`, `meanflow.mode`, `meanflow.cfg_scale`.

Training logs are saved to `logs/{run_name}/`:

```
logs/{run_name}/
├── config.py
├── train.log      # loss, FM/MF loss, MF_V_MSE, grad norm, LR
├── images/
└── ckpts/
```

## Examples

**MNIST** — 10k steps, 1-step sample:

![MNIST](assets/mnist_10k.png)

**MNIST** — 6k steps, 1-step CFG (w=2.0):

![MNIST-cfg](assets/mnist_6k_cfg2.png)

**CIFAR-10** — 200k steps, 1-step CFG (w=2.0):

![CIFAR-10-cfg](assets/cfg_200k_cfg2.png)

## TODO

- [x] Basic training and inference
- [x] Multi-GPU via Accelerate
- [x] Classifier-Free Guidance
- [x] Latent image training
- [x] Improved MeanFlow (iMF)
- [ ] Triton JVP + Flash Attention

## Known Issues

- `jvp` is currently incompatible with PyTorch's native Flash Attention (`scaled_dot_product_attention`).

- Solution in this repo:
  - Run the JVP pass under `no_grad` (for `dudt` only) without Flash Attention.
  - Run a separate gradient-enabled forward pass for optimization with Flash Attention.

- Other advanced solutions:
  - Triton-based kernels with JVP support (e.g. [rcm](https://github.com/NVlabs/rcm)) will allow Flash Attention within JVP.
  - Such kernels may offer additional speed and memory benefits, but are not currently supported in this repository.

## Acknowledgement

Building upon [Just-a-DiT](https://github.com/ArchiMickey/Just-a-DiT) and [EzAudio](https://github.com/haidog-yaqub/EzAudio).

iMF support is based on the official JAX repo [imeanflow](https://github.com/Lyy-iiis/imeanflow). Thanks to the authors for releasing their code.

## Research projects built on this repository

- [MeanAudio](https://github.com/xiquan-li/MeanAudio): fast text-to-audio generation with Mean Flows. See [the paper](https://arxiv.org/abs/2508.06098).
- [MeanVC](https://github.com/ASLP-lab/MeanVC): lightweight streaming zero-shot voice conversion. See [the paper](https://arxiv.org/abs/2510.08392).
- [MeanFlow-TSE](https://github.com/rikishimizu/MeanFlow-TSE): one-step target speaker extraction. See [the paper](https://arxiv.org/abs/2512.18572).

## Like This Project?

If you find this repo helpful, consider dropping a ⭐, it really helps!

## Citation

If you find this repository useful in your research or projects, please consider citing the original MeanFlow papers as well as this implementation.

### MeanFlow

```bibtex
@article{geng2025meanflow,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Shechtman, Eli and Kolter, J. Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```

### Improved MeanFlow

```bibtex
@article{geng2025improved,
  title={Improved Mean Flows: On the Challenges of Fastforward Generative Models},
  author={Geng, Zhengyang and Lu, Yiyang and Wu, Zongze and Shechtman, Eli and Kolter, J. Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2512.02012},
  year={2025}
}
```

### This Repository

```bibtex
@misc{meanflow_pytorch,
  title={MeanFlow: Unofficial PyTorch Implementation},
  author={haidog-yaqub},
  year={2025},
  howpublished={\url{https://github.com/haidog-yaqub/MeanFlow}},
}
```
