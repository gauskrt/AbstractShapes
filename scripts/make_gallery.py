"""Generate the images used by the GitHub Pages site (docs/index.html).

    python scripts/make_gallery.py docs/images/painting.jpg

Writes into docs/images/:
  original.jpg   the input, resized for the web
  main.svg       the step-by-step reconstruction with triangles (for the slider)
  <mode>.svg     one reconstruction per shape vocabulary
  meta.json      dimensions, shape counts and the error after every step
"""

import argparse
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abstractshapes import Model  # noqa: E402

GALLERY = [
    ("triangles", 1),
    ("ellipses", 3),
    ("rectangles", 5),
    ("beziers", 6),
    ("combo", 0),
]


def run(img, mode, count, workers, seed, log=None):
    model = Model(img, workers=workers, seed=seed)
    scores = [round(model.score(), 6)]
    t0 = time.time()
    try:
        for i in range(count):
            scores.append(round(model.step(mode=mode, alpha=128, tries=max(8, workers)), 6))
            if log and (i + 1) % 25 == 0:
                print(f"  {log}: {i + 1}/{count} shapes, score {scores[-1]:.4f}, {time.time() - t0:.0f}s", flush=True)
    finally:
        model.close()
    return model, scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    p.add_argument("-n", type=int, default=500, help="shapes in the main reconstruction")
    p.add_argument("--gallery-n", type=int, default=200, help="shapes per gallery image")
    p.add_argument("-r", type=int, default=256, help="working resolution")
    p.add_argument("-j", type=int, default=os.cpu_count() or 1, help="workers")
    p.add_argument("--seed", type=int, default=1848)  # the year Raja Ravi Varma was born
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = Image.open(args.input).convert("RGB")

    web = src.copy()
    web.thumbnail((900, 900), Image.LANCZOS)
    web.save(os.path.join(args.out, "original.jpg"), quality=88)

    work = src.copy()
    work.thumbnail((args.r, args.r), Image.LANCZOS)

    print(f"main: {args.n} triangles at {work.size[0]}x{work.size[1]}", flush=True)
    model, scores = run(work, 1, args.n, args.j, args.seed, log="main")
    with open(os.path.join(args.out, "main.svg"), "w") as f:
        f.write(model.svg(size=900))

    gallery = []
    for name, mode in GALLERY:
        print(f"gallery: {name}", flush=True)
        m, s = run(work, mode, args.gallery_n, args.j, args.seed, log=name)
        with open(os.path.join(args.out, f"{name}.svg"), "w") as f:
            f.write(m.svg(size=600))
        gallery.append({"name": name, "mode": mode, "shapes": args.gallery_n, "score": s[-1]})

    meta = {
        "width": model.w,
        "height": model.h,
        "shapes": args.n,
        "background": "#%02x%02x%02x" % model.background,
        "scores": scores,
        "gallery": gallery,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
