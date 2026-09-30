"""The optimization model: repeatedly finds the single shape that, when drawn
on top of the current canvas, reduces the error to the target image the most.

For each new shape we:
  1. generate ``candidates`` random shapes and keep the best one,
  2. hill-climb it with random mutations until ``age`` mutations in a row
     fail to improve it,
  3. repeat steps 1-2 ``tries`` times (optionally in parallel) and keep the
     overall winner.

For any shape and opacity, the colour that minimizes the error is computed
directly (it is just a mean over the covered pixels), so only the geometry
needs to be searched.
"""

import math
import multiprocessing as mp

import numpy as np
from PIL import Image

from .shapes import random_shape

# Per-process state for worker processes.
_TARGET = None


def _init_worker(target):
    global _TARGET
    _TARGET = target


def optimal_color(target, current, mask_bool, x0, y0, alpha):
    """Return the RGB colour (ints) that best fits ``target`` when blended with
    ``alpha`` over ``current`` inside the mask, along with the pixel slices."""
    h, w = mask_bool.shape
    t = target[y0:y0 + h, x0:x0 + w][mask_bool]
    c = current[y0:y0 + h, x0:x0 + w][mask_bool]
    a = alpha / 255.0
    color = np.clip(np.rint((t - c * (1.0 - a)).mean(axis=0) / a), 0, 255)
    return color, t, c


def energy(target, current, sse, shape):
    """Total squared error of the canvas if ``shape`` were added."""
    h, w = current.shape[:2]
    r = shape.rasterize(w, h)
    if r is None:
        return sse, None
    x0, y0, mask = r
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return sse, None
    color, t, c = optimal_color(target, current, mask_bool, x0, y0, shape.alpha)
    a = shape.alpha / 255.0
    new = c + (color - c) * a
    delta = np.square(new - t).sum() - np.square(c - t).sum()
    return sse + float(delta), color


def _new_shape(mode, alpha, w, h, rng):
    s = random_shape(mode, w, h, rng)
    s.alpha = alpha if alpha else 128
    return s


def _mutate(shape, optimize_alpha, rng):
    s = shape.mutated()
    if optimize_alpha:
        s.alpha = int(np.clip(s.alpha + rng.integers(-10, 11), 1, 255))
    return s


def search(target, current, sse, mode, alpha, candidates, age, seed):
    """One random-restart hill climb. Returns ``(energy, shape)``."""
    rng = np.random.default_rng(seed)
    h, w = current.shape[:2]

    best, best_e = None, math.inf
    for _ in range(candidates):
        s = _new_shape(mode, alpha, w, h, rng)
        e, _ = energy(target, current, sse, s)
        if e < best_e:
            best, best_e = s, e

    fails = 0
    while fails < age:
        s = _mutate(best, alpha == 0, rng)
        e, _ = energy(target, current, sse, s)
        if e < best_e:
            best, best_e = s, e
            fails = 0
        else:
            fails += 1
    return best_e, best


def _search_worker(args):
    current, sse, mode, alpha, candidates, age, seed = args
    e, shape = search(_TARGET, current, sse, mode, alpha, candidates, age, seed)
    shape.rng = None  # don't ship the generator back
    return e, shape


class Model:
    def __init__(self, target_image, background=None, workers=1, seed=None):
        img = target_image.convert("RGB")
        self.w, self.h = img.size
        self.target = np.asarray(img, dtype=np.float64)
        if background is None:
            background = tuple(int(round(v)) for v in self.target.reshape(-1, 3).mean(axis=0))
        self.background = tuple(background)
        self.current = np.empty_like(self.target)
        self.current[:] = self.background
        self.sse = float(np.square(self.current - self.target).sum())
        self.shapes = []  # list of (shape, color, alpha)
        self.rng = np.random.default_rng(seed)
        self.workers = max(1, workers)
        self.pool = None
        if self.workers > 1:
            self.pool = mp.Pool(self.workers, initializer=_init_worker, initargs=(self.target,))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def score(self):
        """Root-mean-square error, normalized to [0, 1]."""
        return math.sqrt(self.sse / (self.w * self.h * 3)) / 255.0

    def step(self, mode=1, alpha=128, candidates=200, age=100, tries=8):
        """Find and add one shape. Returns the new score."""
        seeds = self.rng.integers(0, 2**63 - 1, size=tries)
        if self.pool is not None:
            jobs = [(self.current, self.sse, mode, alpha, candidates, age, int(s)) for s in seeds]
            results = self.pool.map(_search_worker, jobs)
        else:
            results = [
                search(self.target, self.current, self.sse, mode, alpha, candidates, age, int(s))
                for s in seeds
            ]
        _, best = min(results, key=lambda r: r[0])
        best.rng = self.rng
        self.add(best)
        return self.score()

    def add(self, shape):
        r = shape.rasterize(self.w, self.h)
        if r is None:
            return
        x0, y0, mask = r
        mask_bool = np.asarray(mask) > 0
        if not mask_bool.any():
            return
        color, _, c = optimal_color(self.target, self.current, mask_bool, x0, y0, shape.alpha)
        a = shape.alpha / 255.0
        mh, mw = mask_bool.shape
        region = self.current[y0:y0 + mh, x0:x0 + mw]
        t = self.target[y0:y0 + mh, x0:x0 + mw][mask_bool]
        new = c + (color - c) * a
        self.sse += float(np.square(new - t).sum() - np.square(c - t).sum())
        region[mask_bool] = new
        self.shapes.append((shape, tuple(int(v) for v in color), shape.alpha))

    # -- output ------------------------------------------------------------
    def render(self, size=None, count=None):
        """Render the first ``count`` shapes so the longest side is ``size``."""
        scale = 1.0 if size is None else size / max(self.w, self.h)
        ow, oh = max(1, round(self.w * scale)), max(1, round(self.h * scale))
        img = Image.new("RGB", (ow, oh), self.background)
        for shape, color, alpha in self.shapes[:count]:
            r = shape.rasterize(ow, oh, scale=scale, value=alpha)
            if r is None:
                continue
            x0, y0, mask = r
            img.paste(color, (x0, y0, x0 + mask.width, y0 + mask.height), mask)
        return img

    def svg(self, size=None, count=None):
        scale = 1.0 if size is None else size / max(self.w, self.h)
        ow, oh = round(self.w * scale), round(self.h * scale)
        bg = "#%02x%02x%02x" % self.background
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{ow}" height="{oh}">',
            f'<rect x="0" y="0" width="{ow}" height="{oh}" fill="{bg}" />',
            f'<g transform="scale({scale:.6f}) translate(0.5 0.5)">',
        ]
        for shape, color, alpha in self.shapes[:count]:
            attrs = 'fill="#%02x%02x%02x" fill-opacity="%.4f"' % (*color, alpha / 255.0)
            lines.append(shape.svg(attrs))
        lines.append("</g>")
        lines.append("</svg>")
        return "\n".join(lines) + "\n"
