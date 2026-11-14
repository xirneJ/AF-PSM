# AF-PSM on MNIST, short-run config (10k steps) for validating the method on a
# shared GPU before committing to the 200k-step runs.
#
# Memory-friendly defaults: bf16 + memory-efficient attention + gradient
# checkpointing.  Override anything from the CLI without editing this file:
#   python train.py --config configs/afpsm_mnist_10k.py --batch_size 16 --flash 1 --ckpt 1
#
# The alpha curriculum uses n_steps, so keep n_steps at the real run length.
config = dict(
    name="afpsm_mnist_10k",
    dataset="mnist",
    data_root="mnist",
    n_steps=10_000,
    batch_size=16,
    num_workers=8,
    image_size=32,
    lr=1e-4,
    weight_decay=0.0,
    grad_clip=5.0,
    mixed_precision='bf16',
    scheduler=dict(warmup_steps=500, min_lr_ratio=0.1),
    log_step=100,
    sample_step=500,
    sample_nrow=8,
    sample_classes=[0, 1, 2, 3, 4, 5, 6, 7],
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
        output_activation='tanh',
        alpha_min=0.01,
        grad_checkpoint=True,
    ),
    meanflow=dict(
        mode="af-psm",
        normalizer=["minmax", None, None],
        time_dist=["lognorm", -0.4, 1.0],
        alpha_min=0.01,
        alpha_curriculum="linear",      # try "to_half" for the ablation
        alpha_weight=0.1,
        freq_weight=0.5,
        mid_weight=0.5,
        alpha_stopgrad=True,
        detach_xs=False,
        use_flash_attention=True,       # memory-efficient SDPA in loss + sampler
        freq_loss="rel_hp",
        freq_gamma=1.0,
    ),
)
