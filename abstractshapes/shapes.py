"""Shape primitives that can be randomly generated, mutated and rasterized.

Every shape knows how to:
  * create a random instance of itself (``random``)
  * make a small random change to itself (``mutate``)
  * rasterize itself into a mask (``rasterize``) at any scale
  * describe itself as an SVG element (``svg``)

Rasterization is delegated to Pillow's ImageDraw so that the exact same code
is used while optimizing (at the working resolution) and when rendering the
final output (at the output resolution).
"""

import math

import numpy as np
from PIL import Image, ImageDraw

# How far outside the image a shape's control points are allowed to wander.
MARGIN = 16


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _rotate(points, angle_deg, cx, cy):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in points]


class Shape:
    """Base class. Subclasses implement ``_random``, ``mutate``, ``_draw``,
    ``bounds`` and ``svg``."""

    name = "shape"

    def __init__(self, w, h, rng):
        self.w = w
        self.h = h
        self.rng = rng

    @classmethod
    def random(cls, w, h, rng):
        s = cls(w, h, rng)
        s._random()
        while not s.valid():
            s._random()
        return s

    def copy(self):
        c = self.__class__.__new__(self.__class__)
        c.__dict__.update(self.__dict__)
        return c

    def valid(self):
        return True

    def mutated(self):
        """Return a mutated, valid copy of this shape."""
        while True:
            c = self.copy()
            c.mutate()
            if c.valid():
                return c

    # -- helpers -----------------------------------------------------------
    def _rand_point(self):
        return (float(self.rng.integers(0, self.w)), float(self.rng.integers(0, self.h)))

    def _near(self, x, y, spread=15):
        return (
            x + float(self.rng.integers(-spread, spread + 1)),
            y + float(self.rng.integers(-spread, spread + 1)),
        )

    def _jitter_point(self, x, y, amount=16.0):
        x = _clamp(x + self.rng.normal() * amount, -MARGIN, self.w - 1 + MARGIN)
        y = _clamp(y + self.rng.normal() * amount, -MARGIN, self.h - 1 + MARGIN)
        return x, y

    # -- rasterization -----------------------------------------------------
    def rasterize(self, w, h, scale=1.0, value=255):
        """Rasterize the shape into an image of size ``w`` x ``h`` where the
        shape's coordinates are multiplied by ``scale``.

        Returns ``(x0, y0, mask)`` where ``mask`` is an 8-bit Pillow image
        covering the clipped bounding box whose top-left corner is at
        ``(x0, y0)``, or ``None`` if the shape lies entirely off-canvas.
        """
        bx0, by0, bx1, by1 = self.bounds()
        pad = 2 + self.pad() * scale
        x0 = max(0, int(math.floor(bx0 * scale - pad)))
        y0 = max(0, int(math.floor(by0 * scale - pad)))
        x1 = min(w, int(math.ceil(bx1 * scale + pad)) + 1)
        y1 = min(h, int(math.ceil(by1 * scale + pad)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        self._draw(ImageDraw.Draw(mask), scale, x0, y0, value)
        return x0, y0, mask

    def pad(self):
        return 0

    @staticmethod
    def _xf(points, scale, ox, oy):
        return [(x * scale - ox, y * scale - oy) for x, y in points]


class PolygonShape(Shape):
    """A shape that is drawn as a filled polygon given by ``points()``."""

    def points(self):
        raise NotImplementedError

    def bounds(self):
        pts = self.points()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def _draw(self, draw, scale, ox, oy, value):
        draw.polygon(self._xf(self.points(), scale, ox, oy), fill=value)

    def svg(self, attrs):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in self.points())
        return f'<polygon {attrs} points="{pts}" />'


class Triangle(PolygonShape):
    name = "triangle"

    def _random(self):
        x1, y1 = self._rand_point()
        x2, y2 = self._near(x1, y1)
        x3, y3 = self._near(x1, y1)
        self.pts = [(x1, y1), (x2, y2), (x3, y3)]

    def points(self):
        return self.pts

    def mutate(self):
        while True:
            pts = list(self.pts)
            i = int(self.rng.integers(0, 3))
            pts[i] = self._jitter_point(*pts[i])
            old, self.pts = self.pts, pts
            if self.valid():
                return
            self.pts = old

    def valid(self):
        # Reject slivers: every interior angle must be at least 15 degrees.
        (x1, y1), (x2, y2), (x3, y3) = self.pts
        min_deg = 15.0

        def angle(ax, ay, bx, by):
            la, lb = math.hypot(ax, ay), math.hypot(bx, by)
            if la == 0 or lb == 0:
                return 0.0
            d = _clamp((ax * bx + ay * by) / (la * lb), -1.0, 1.0)
            return math.degrees(math.acos(d))

        a1 = angle(x2 - x1, y2 - y1, x3 - x1, y3 - y1)
        a2 = angle(x1 - x2, y1 - y2, x3 - x2, y3 - y2)
        a3 = 180.0 - a1 - a2
        return a1 > min_deg and a2 > min_deg and a3 > min_deg


class Rectangle(Shape):
    name = "rectangle"

    def _random(self):
        x1, y1 = self._rand_point()
        x2 = _clamp(x1 + float(self.rng.integers(0, 32)), 0, self.w - 1)
        y2 = _clamp(y1 + float(self.rng.integers(0, 32)), 0, self.h - 1)
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def mutate(self):
        if self.rng.integers(0, 2) == 0:
            self.x1, self.y1 = self._jitter_point(self.x1, self.y1)
        else:
            self.x2, self.y2 = self._jitter_point(self.x2, self.y2)

    def bounds(self):
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    def _draw(self, draw, scale, ox, oy, value):
        x0, y0, x1, y1 = self.bounds()
        # Scaled so that an integer rectangle covers whole output pixels.
        draw.rectangle(
            [x0 * scale - ox, y0 * scale - oy, (x1 + 1) * scale - ox - 1, (y1 + 1) * scale - oy - 1],
            fill=value,
        )

    def svg(self, attrs):
        x0, y0, x1, y1 = self.bounds()
        return (
            f'<rect {attrs} x="{x0:.2f}" y="{y0:.2f}" '
            f'width="{x1 - x0 + 1:.2f}" height="{y1 - y0 + 1:.2f}" />'
        )


class RotatedRectangle(PolygonShape):
    name = "rotated rectangle"

    def _random(self):
        self.x, self.y = self._rand_point()
        self.sx = float(self.rng.integers(1, 33))
        self.sy = float(self.rng.integers(1, 33))
        self.angle = float(self.rng.integers(0, 360))

    def mutate(self):
        r = self.rng.integers(0, 3)
        if r == 0:
            self.x, self.y = self._jitter_point(self.x, self.y)
        elif r == 1:
            self.sx = _clamp(self.sx + self.rng.normal() * 16, 1, self.w - 1)
            self.sy = _clamp(self.sy + self.rng.normal() * 16, 1, self.h - 1)
        else:
            self.angle = self.angle + self.rng.normal() * 32

    def valid(self):
        a, b = max(self.sx, self.sy), min(self.sx, self.sy)
        return a / b <= 5

    def points(self):
        hx, hy = self.sx / 2, self.sy / 2
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        return _rotate(corners, self.angle, self.x, self.y)


class Ellipse(Shape):
    name = "ellipse"
    circle = False

    def _random(self):
        self.x, self.y = self._rand_point()
        self.rx = float(self.rng.integers(1, 33))
        self.ry = self.rx if self.circle else float(self.rng.integers(1, 33))

    def mutate(self):
        r = self.rng.integers(0, 3)
        if r == 0:
            self.x, self.y = self._jitter_point(self.x, self.y)
        elif r == 1 or self.circle:
            self.rx = _clamp(self.rx + self.rng.normal() * 16, 1, self.w - 1)
            if self.circle:
                self.ry = self.rx
        else:
            self.ry = _clamp(self.ry + self.rng.normal() * 16, 1, self.h - 1)

    def bounds(self):
        return self.x - self.rx, self.y - self.ry, self.x + self.rx, self.y + self.ry

    def _draw(self, draw, scale, ox, oy, value):
        x0, y0, x1, y1 = self.bounds()
        draw.ellipse(
            [x0 * scale - ox, y0 * scale - oy, x1 * scale - ox, y1 * scale - oy],
            fill=value,
        )

    def svg(self, attrs):
        return (
            f'<ellipse {attrs} cx="{self.x:.2f}" cy="{self.y:.2f}" '
            f'rx="{self.rx:.2f}" ry="{self.ry:.2f}" />'
        )


class Circle(Ellipse):
    name = "circle"
    circle = True


class RotatedEllipse(PolygonShape):
    name = "rotated ellipse"
    segments = 32

    def _random(self):
        self.x, self.y = self._rand_point()
        self.rx = float(self.rng.integers(1, 33))
        self.ry = float(self.rng.integers(1, 33))
        self.angle = float(self.rng.integers(0, 360))

    def mutate(self):
        r = self.rng.integers(0, 3)
        if r == 0:
            self.x, self.y = self._jitter_point(self.x, self.y)
        elif r == 1:
            self.rx = _clamp(self.rx + self.rng.normal() * 16, 1, self.w - 1)
            self.ry = _clamp(self.ry + self.rng.normal() * 16, 1, self.w - 1)
        else:
            self.angle = self.angle + self.rng.normal() * 32

    def points(self):
        n = self.segments
        pts = [
            (self.rx * math.cos(2 * math.pi * i / n), self.ry * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
        return _rotate(pts, self.angle, self.x, self.y)

    def svg(self, attrs):
        return (
            f'<g transform="translate({self.x:.2f} {self.y:.2f}) rotate({self.angle:.2f})">'
            f'<ellipse {attrs} cx="0" cy="0" rx="{self.rx:.2f}" ry="{self.ry:.2f}" /></g>'
        )


class Quadratic(Shape):
    """A quadratic Bezier curve stroked with a (mutable) line width."""

    name = "bezier"
    steps = 24

    def _random(self):
        x1, y1 = self._rand_point()
        x2, y2 = self._near(x1, y1, 20)
        x3, y3 = self._near(x2, y2, 20)
        self.pts = [(x1, y1), (x2, y2), (x3, y3)]
        self.width = 1.0

    def mutate(self):
        while True:
            pts = list(self.pts)
            r = self.rng.integers(0, 4)
            if r < 3:
                pts[r] = self._jitter_point(*pts[r], amount=8)
            else:
                self.width = _clamp(self.width + self.rng.normal(), 1, 16)
            old, self.pts = self.pts, pts
            if self.valid():
                return
            self.pts = old

    def valid(self):
        (x1, y1), (x2, y2), (x3, y3) = self.pts
        dx12, dy12 = x1 - x2, y1 - y2
        dx23, dy23 = x2 - x3, y2 - y3
        dx13, dy13 = x1 - x3, y1 - y3
        d12 = dx12 * dx12 + dy12 * dy12
        d23 = dx23 * dx23 + dy23 * dy23
        d13 = dx13 * dx13 + dy13 * dy13
        return d13 > d12 and d13 > d23

    def curve(self):
        (x1, y1), (x2, y2), (x3, y3) = self.pts
        out = []
        for i in range(self.steps + 1):
            t = i / self.steps
            u = 1 - t
            out.append((u * u * x1 + 2 * u * t * x2 + t * t * x3, u * u * y1 + 2 * u * t * y2 + t * t * y3))
        return out

    def bounds(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)

    def pad(self):
        return self.width

    def _draw(self, draw, scale, ox, oy, value):
        width = max(1, int(round(self.width * scale)))
        draw.line(self._xf(self.curve(), scale, ox, oy), fill=value, width=width, joint="curve")

    def svg(self, attrs):
        (x1, y1), (x2, y2), (x3, y3) = self.pts
        attrs = attrs.replace("fill=", "stroke=").replace("fill-opacity=", "stroke-opacity=")
        return (
            f'<path fill="none" {attrs} stroke-width="{self.width:.2f}" '
            f'd="M {x1:.2f} {y1:.2f} Q {x2:.2f} {y2:.2f}, {x3:.2f} {y3:.2f}" />'
        )


class Polygon(PolygonShape):
    """A free-form quadrilateral."""

    name = "polygon"
    order = 4

    def _random(self):
        x, y = self._rand_point()
        self.pts = [(x, y)] + [self._near(x, y, 20) for _ in range(self.order - 1)]

    def points(self):
        return self.pts

    def mutate(self):
        while True:
            pts = list(self.pts)
            if self.rng.random() < 0.25:
                i, j = self.rng.integers(0, self.order, size=2)
                pts[i], pts[j] = pts[j], pts[i]
            else:
                i = int(self.rng.integers(0, self.order))
                pts[i] = self._jitter_point(*pts[i])
            old, self.pts = self.pts, pts
            if self.valid():
                return
            self.pts = old

    def valid(self):
        # Require a convex polygon (all cross products share the same sign).
        sign = 0
        n = len(self.pts)
        for i in range(n):
            x1, y1 = self.pts[i]
            x2, y2 = self.pts[(i + 1) % n]
            x3, y3 = self.pts[(i + 2) % n]
            cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            s = 1 if cross > 0 else -1 if cross < 0 else 0
            if s == 0:
                return False
            if sign == 0:
                sign = s
            elif s != sign:
                return False
        return True


# Mode numbers match the original `primitive` tool.
MODES = {
    1: Triangle,
    2: Rectangle,
    3: Ellipse,
    4: Circle,
    5: RotatedRectangle,
    6: Quadratic,
    7: RotatedEllipse,
    8: Polygon,
}

MODE_NAMES = {
    0: "combo",
    1: "triangle",
    2: "rectangle",
    3: "ellipse",
    4: "circle",
    5: "rotated rectangle",
    6: "beziers",
    7: "rotated ellipse",
    8: "polygon",
}


def random_shape(mode, w, h, rng):
    """Create a random shape for the given mode (0 = any of 1..5,7,8)."""
    if mode == 0:
        # Combo mode mirrors primitive: every filled shape type.
        choices = (1, 2, 3, 4, 5, 7, 8)
        mode = choices[int(rng.integers(0, len(choices)))]
    return MODES[mode].random(w, h, rng)
