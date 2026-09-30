"""Command line interface, modelled on the original `primitive` flags.

    python -m abstractshapes -i input.png -o output.png -n 100
"""

import argparse
import os
import sys
import time

from PIL import Image

from .model import Model
from .shapes import MODE_NAMES


def parse_color(s):
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise argparse.ArgumentTypeError(f"invalid hex color: {s!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def build_parser():
    modes = ", ".join(f"{k}={v}" for k, v in MODE_NAMES.items())
    p = argparse.ArgumentParser(
        prog="abstractshapes",
        description="Reproduce an image using geometric primitives (a Python port of fogleman/primitive).",
    )
    p.add_argument("-i", dest="input", required=True, help="input image path")
    p.add_argument(
        "-o",
        dest="outputs",
        action="append",
        required=True,
        help="output path (.png, .jpg, .svg or .gif); may be repeated. "
        "Include a %%d / %%03d placeholder to save every frame",
    )
    p.add_argument("-n", dest="count", type=int, required=True, help="number of shapes")
    p.add_argument("-m", dest="mode", type=int, default=1, choices=sorted(MODE_NAMES), help=f"mode: {modes} (default 1)")
    p.add_argument("-a", dest="alpha", type=int, default=128, help="alpha 1-255, or 0 to let the algorithm choose (default 128)")
    p.add_argument("-r", dest="resize", type=int, default=256, help="resize the input so its longest side is this (default 256)")
    p.add_argument("-s", dest="size", type=int, default=1024, help="output image size (default 1024)")
    p.add_argument("-bg", dest="background", type=parse_color, default=None, help="background color as hex (default: average color)")
    p.add_argument("-j", dest="workers", type=int, default=os.cpu_count() or 1, help="parallel workers (default: all cores)")
    p.add_argument("-nth", dest="nth", type=int, default=1, help="save every Nth frame (only with %%d in output, default 1)")
    p.add_argument("--candidates", type=int, default=200, help="random shapes tried per hill climb (default 200)")
    p.add_argument("--age", type=int, default=100, help="failed mutations before a hill climb stops (default 100)")
    p.add_argument("--tries", type=int, default=None, help="hill climbs per shape (default: max(8, workers))")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")
    p.add_argument("-v", dest="verbose", action="store_true", help="verbose output")
    return p


def save(model, path, size, frames=None, count=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        with open(path, "w") as f:
            f.write(model.svg(size, count))
    elif ext == ".gif":
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    elif ext in (".png", ".jpg", ".jpeg"):
        model.render(size, count).save(path)
    else:
        raise SystemExit(f"unsupported output format: {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (0 <= args.alpha <= 255):
        raise SystemExit("alpha must be between 0 and 255")

    img = Image.open(args.input).convert("RGB")
    if args.resize > 0:
        img.thumbnail((args.resize, args.resize), Image.LANCZOS)

    tries = args.tries or max(8, args.workers)
    model = Model(img, background=args.background, workers=args.workers, seed=args.seed)

    wants_gif = any(o.lower().endswith(".gif") for o in args.outputs)
    frames = [model.render(min(args.size, 512))] if wants_gif else None

    if args.verbose:
        print(f"{args.input}: {model.w}x{model.h}, mode {MODE_NAMES[args.mode]}, {args.count} shapes, {args.workers} worker(s)")

    start = time.time()
    try:
        for i in range(1, args.count + 1):
            t0 = time.time()
            score = model.step(args.mode, args.alpha, args.candidates, args.age, tries)
            if args.verbose:
                print(f"{i}: t={time.time() - start:.1f}s, step={time.time() - t0:.2f}s, score={score:.6f}")
            if wants_gif:
                frames.append(model.render(min(args.size, 512)))
            last = i == args.count
            for out in args.outputs:
                if "%" in out:
                    if i % args.nth == 0 or last:
                        save(model, out % i, args.size, frames)
                elif last and not out.lower().endswith(".gif"):
                    save(model, out, args.size)
                elif last:
                    save(model, out, args.size, frames)
    except KeyboardInterrupt:
        print("\ninterrupted, saving what we have...", file=sys.stderr)
        for out in args.outputs:
            if "%" not in out and model.shapes:
                save(model, out, args.size, frames)
    finally:
        model.close()


if __name__ == "__main__":
    main()
