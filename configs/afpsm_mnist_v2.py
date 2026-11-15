# AF-PSM v2 on MNIST: JVP-free consistency distillation + CFG.
#
#   M1  only the two-leg composition is anchored to data; the direct 1-step
#       prediction is distilled from it (stopgrad teacher)
#   M2  mixed logit-normal / uniform time sampling, so the inference endpoints
#       (t=1, r=0) are actually covered by training
#   M3  CFG distilled into the w input (same recipe as the iMF baseline), so
#       guidance costs no extra NFE at sampling
#   M4  no alpha head / curriculum: the split point is random (every s in (r,t)
#       is an equally valid training interval); the vacuous freq/mid terms are off
#
# Quick check:  --n_steps 10000   (about 1.8 h on a shared 5090)
# Official:     --n_steps 30000   (matched against the iMF baseline's 960k samples)
config = dict(
    name="afpsm_mnist_v2",
    dataset="mnist",
    data_root="data/mnist",
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
        in_channels=1,
        dim=512,
        depth=12,
        num_heads=8,
        num_classes=10,
        max_cfg=7.5,
        output_mode='x',
        # unbounded output; tanh saturates on the minmax +-1 targets
        output_activation='none',
        alpha_min=0.01,
    ),
    meanflow=dict(
        mode="af-psm",
        normalizer=["minmax", None, None],
        # M2: 50% uniform -> endpoints (t ~ 1, r ~ 0) get real training mass
        time_dist=["mixed", -0.4, 1.0, 0.5],
        alpha_min=0.01,
        # M4: no learned alpha / no curriculum -- random split in [0.1, 0.9]
        alpha_mode="random",
        split_min=0.1,
        # M1: anchor the composition, distil it into the direct 1-step path
        pixel_weight=0.0,
        anchor_weight=1.0,
        cons_weight=1.0,
        # M3: CFG distillation (identical distribution to the iMF baseline)
        cfg_ratio=0.1,
        cfg_scale=[1.0, 7.5],
        cfg_scale_ratio=0.75,
        # retired v1 terms, kept as knobs for the ablation table
        freq_weight=0.0,
        mid_weight=0.0,
        alpha_weight=0.0,
        alpha_stopgrad=True,
        detach_xs=False,
        freq_loss="rel_hp",
        freq_gamma=1.0,
    ),
)
