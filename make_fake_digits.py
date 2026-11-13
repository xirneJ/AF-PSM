"""Generate a tiny offline digit dataset (ImageFolder layout) for CPU smoke runs.

The MNIST mirrors are not always reachable; this produces a class-conditional
32x32 digit dataset that lets `configs/smoke_cpu.py` exercise the full training
loop (dataloader -> AF-PSM loss -> logging -> sampling -> checkpoint) offline.

Usage:
    python make_fake_digits.py                # -> fakedigits/<0..9>/*.png
    python make_fake_digits.py --root mydir   # custom output root
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='fakedigits')
    parser.add_argument('--n_per_class', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--font', type=str, default='C:/Windows/Fonts/arial.ttf')
    args = parser.parse_args()

    try:
        font = ImageFont.truetype(args.font, size=int(args.image_size * 0.9))
    except OSError:
        font = ImageFont.load_default()

    for label in range(10):
        out_dir = os.path.join(args.root, str(label))
        os.makedirs(out_dir, exist_ok=True)
        for i in range(args.n_per_class):
            img = Image.new('L', (args.image_size, args.image_size), color=255)
            draw = ImageDraw.Draw(img)
            # tiny jitter so the 8 variants per class are not identical
            dx = (i % 3) - 1
            dy = ((i // 3) % 3) - 1
            draw.text(
                (args.image_size / 2 + dx, args.image_size / 2 + dy),
                str(label),
                fill=0,
                font=font,
                anchor='mm',
            )
            img.save(os.path.join(out_dir, f'img_{i:02d}.png'))

    print(f"wrote {args.root}/<0..9>/*.png "
          f"({10 * args.n_per_class} images, {args.image_size}x{args.image_size})")


if __name__ == '__main__':
    main()
