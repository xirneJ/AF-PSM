import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

# All datasets live under <repo root>/data by default, resolved against the
# repository root (NOT the current working directory), so training can be
# launched from anywhere:
#     <repo root>/data/mnist/MNIST/...
#     <repo root>/data/cifar/cifar-10-batches-py/...
#     <repo root>/data/imagenet/<class>/...
# Set AFPSM_DATA_ROOT to move the dataset tree to another disk; relative
# data_root paths are then resolved under it instead of the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def resolve_data_root(cfg):
    """Resolve a config data_root to an absolute path (repo-root anchored)."""
    root = Path(cfg.get("data_root", "data"))
    if root.is_absolute():
        return root
    env_root = os.environ.get("AFPSM_DATA_ROOT")
    base = Path(env_root).expanduser().resolve() if env_root else PROJECT_ROOT
    return (base / root).resolve()


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_dataset(cfg):
    root = resolve_data_root(cfg)

    if cfg["dataset"] == "cifar10":
        root.mkdir(parents=True, exist_ok=True)
        return torchvision.datasets.CIFAR10(
            root=str(root),
            train=True,
            download=True,
            transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
        )

    if cfg["dataset"] == "mnist":
        root.mkdir(parents=True, exist_ok=True)
        return torchvision.datasets.MNIST(
            root=str(root),
            train=True,
            download=True,
            transform=T.Compose([
                T.Resize((cfg["image_size"], cfg["image_size"])),
                T.ToTensor(),
            ]),
        )

    if cfg["dataset"] == "imagenet":
        transform = T.Compose([
            T.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg["image_size"])),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        return ImageFolder(str(root), transform=transform)

    if cfg["dataset"] == "folder":
        # generic image folder: keeps raw [0, 1] pixels, so `normalizer=["minmax"]`
        # maps them to [-1, 1] exactly like the MNIST branch.
        # NOTE: torchvision's default loader forces RGB; keep the original mode
        # instead so that grayscale folders stay single-channel.
        transform = T.Compose([
            T.Resize((cfg["image_size"], cfg["image_size"])),
            T.ToTensor(),
        ])

        def _keep_mode_loader(path):
            with open(path, "rb") as f:
                img = Image.open(f)
                img.load()
            return img

        return ImageFolder(str(root), transform=transform,
                           loader=_keep_mode_loader)

    raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def build_eval_dataset(cfg, train=False):
    """Deterministic evaluation split (no random augmentation); images in [0, 1]."""
    root = resolve_data_root(cfg)

    if cfg["dataset"] == "mnist":
        root.mkdir(parents=True, exist_ok=True)
        return torchvision.datasets.MNIST(
            root=str(root), train=train, download=True,
            transform=T.Compose([
                T.Resize((cfg["image_size"], cfg["image_size"])),
                T.ToTensor(),
            ]),
        )

    if cfg["dataset"] == "cifar10":
        root.mkdir(parents=True, exist_ok=True)
        return torchvision.datasets.CIFAR10(
            root=str(root), train=train, download=True,
            transform=T.Compose([T.ToTensor()]),
        )

    if cfg["dataset"] in ("imagenet", "folder"):
        transform = T.Compose([
            T.Resize((cfg["image_size"], cfg["image_size"])),
            T.ToTensor(),
        ])

        def _keep_mode_loader(path):
            with open(path, "rb") as f:
                img = Image.open(f)
                img.load()
            return img

        return ImageFolder(str(root), transform=transform, loader=_keep_mode_loader)

    raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def cycle(iterable):
    while True:
        for item in iterable:
            yield item
