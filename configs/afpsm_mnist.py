config = dict(
    name="afpsm_mnist",
    dataset="mnist",
    data_root="mnist",
    n_steps=200_000,
    batch_size=64,
    num_workers=8,
    image_size=32,
    lr=1e-4,
    weight_decay=0.0,
    grad_clip=5.0,
    mixed_precision='bf16',
    scheduler=dict(warmup_steps=2500, min_lr_ratio=0.1),
    log_step=500,
    sample_step=500,
    sample_nrow=10,
    sample_classes=None,
    # AF-PSM sampler: 1 = one-step generation
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
        output_mode='x',          # single clean-image head + alpha head
        output_activation='tanh', # x in [-1, 1]
        alpha_min=0.01,
    ),
    meanflow=dict(
        mode="af-psm",
        normalizer=["minmax", None, None],
        time_dist=["lognorm", -0.4, 1.0],
        alpha_min=0.01,
        # 'linear' = literal curriculum 1 -> alpha_min;
        # alternatives: 'to_half' (1 -> 0.5), 'triangular', 'const', 'none'
        alpha_curriculum="linear",
        alpha_weight=0.1,
        freq_weight=0.5,
        mid_weight=0.5,
        alpha_stopgrad=True,
        detach_xs=False,
        # 'rel_hp' = ortho rfft2 + soft high-pass + energy normalization;
        # 'mse'    = literal MSE on the complex spectrum (dominated by DC, not recommended)
        freq_loss="rel_hp",
        freq_gamma=1.0,
    ),
)
