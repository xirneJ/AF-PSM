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
- the network predicts the clean image directly: `x_hat, alpha = model(z_t, t, r)` with `x_hat = tanh(...)` and `alpha` in `[alpha_min, 1]`
- the dynamic split point is `s = alpha * r + (1 - alpha) * t`; the intermediate sample is reconstructed in closed form from the linearity of the path: `z_s = (1 - s/t) * x_hat_s + (s/t) * z_t`
- 3 shared-parameter forwards, 4 losses (pixel anchoring, frequency consistency, exact midpoint supervision, alpha curriculum), zero JVP

```bash
python train.py --config configs/afpsm_mnist.py                  # MNIST (GPU)
python train.py --config configs/afpsm_cifar10.py                # CIFAR-10 (GPU)
python train.py --config configs/smoke_cpu.py                    # ~400-step CPU sanity check
```

Switches in the `meanflow` block: `alpha_curriculum` (`linear` = 1 -> 0.01, plus `to_half`, `triangular`, `const`, `none`), `freq_loss` (`rel_hp` = ortho rfft2 with soft high-pass weighting and energy normalization, or the literal `mse`), `alpha_weight` / `freq_weight` / `mid_weight`, `alpha_stopgrad`, `detach_xs`.

## Usage

`i-meanflow` (`meanflow.mode: "i-meanflow"`) is more stable and recommended for your projects. `meanflow` mode is kept for reference.

```bash
pip install torch accelerate torchvision einops tqdm diffusers
```

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
