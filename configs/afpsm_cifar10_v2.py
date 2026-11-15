# AF-PSM v2 on CIFAR-10 (same objective as afpsm_mnist_v2.py).
#
#   M1  composition anchored to data, direct 1-step path distilled from it
#   M2  mixed logit-normal / uniform time sampling (covers t=1, r=0)
#   M3  CFG distilled into the w input (no extra NFE at sampling)
#   M4  random split point, no alpha curriculum, freq/mid retired
#
# Quick check:  --n_steps 10000   Short run: --n_steps 30000   Long: --n_steps 200000
config = dict(
    name="afpsm_cifar10_v2",
    dataset="cifar10",
    data_root="data/cifar",
    n_steps=30_000,
    batch_size=32,
    num_workers=8,
    image_size=32,
    lr=1e-4,
    weight_decay=0.0,
    grad_clip=5.0,
    mixed_precision='bf16',
    scheduler=dict(warmup_steps=2500, min_lr_ratio=0.1),
    log_step=100,
    sample_step=500,
    sample_nrow=10,
    sample_classes=None,
    sample_steps=1,
    sample_adaptive=False,
    model=dict(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=512,
        depth=12,
        num_heads=8,
        num_classes=10,
        max_cfg=7.5,
        output_mode='x',
        output_activation='none',
        alpha_min=0.01,
    ),
    meanflow=dict(
        mode="af-psm",
        normalizer=["minmax", None, None],
        time_dist=["mixed", -0.4, 1.0, 0.5],
        alpha_min=0.01,
        alpha_mode="random",
        split_min=0.1,
        pixel_weight=0.0,
        anchor_weight=1.0,
        cons_weight=1.0,
        cfg_ratio=0.1,
        cfg_scale=[1.0, 7.5],
        cfg_scale_ratio=0.75,
        # FID-optimal guidance scale found on MNIST (~2.0); re-sweep on CIFAR
        sample_w=2.0,
        freq_weight=0.0,
        mid_weight=0.0,
        alpha_weight=0.0,
        alpha_stopgrad=True,
        detach_xs=False,
        freq_loss="rel_hp",
        freq_gamma=1.0,
    ),
)
