# AbstractShapes

This is an attempt to make abstract art with Machines.

AbstractShapes reproduces an image using geometric primitives — triangles,
rectangles, ellipses, circles, Bézier curves and polygons. It is a Python
recreation of [**primitive**](https://github.com/fogleman/primitive) by
Michael Fogleman.

## Website

`docs/index.html` is a GitHub Pages site explaining what the project is about and how
it works, using a painting by Raja Ravi Varma as the example input. The
`Deploy site to GitHub Pages` workflow (`.github/workflows/pages.yml`) runs on every
push to `main`: it downloads the painting from Wikimedia Commons, generates the shape
reconstructions with `scripts/make_gallery.py`, and publishes `docs/`.

To turn it on: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
To preview locally:

```bash
python scripts/make_gallery.py path/to/painting.jpg   # writes docs/images/
python -m http.server -d docs
```

## How it works

The goal is to find the one shape that, drawn on top of the current canvas,
brings it closest to the target image. That shape is added, and the process
repeats.

1. Start from a canvas filled with the average color of the target (or a color you choose).
2. Generate a batch of random shapes and keep the one that lowers the error most.
3. Hill-climb it: apply small random mutations (move a vertex, resize, rotate, …)
   and keep each mutation only if it lowers the error. Stop once `--age`
   mutations in a row fail.
4. Run several such climbs (in parallel across CPU cores) and keep the winner.
5. Add the shape to the canvas and go back to step 2.

For a given shape and opacity, the best color has a closed-form solution (a
mean over the covered pixels), so the search only has to explore geometry.
The error metric is the root-mean-square difference between canvas and target.

## Installation

```bash
pip install -r requirements.txt
# or, to get an `abstractshapes` command:
pip install .
```

Requires Python 3.8+, NumPy and Pillow.

## Usage

```bash
python -m abstractshapes -i input.jpg -o output.png -n 100
```

| Flag | Default | Description |
| --- | --- | --- |
| `-i` | *required* | input image |
| `-o` | *required* | output file: `.png`, `.jpg`, `.svg` or `.gif` (repeatable) |
| `-n` | *required* | number of shapes |
| `-m` | 1 | mode: 0=combo 1=triangle 2=rectangle 3=ellipse 4=circle 5=rotated rectangle 6=beziers 7=rotated ellipse 8=polygon |
| `-a` | 128 | shape opacity (1–255); `0` lets the algorithm pick opacity per shape |
| `-r` | 256 | resize input so its longest side is this many pixels (smaller = faster) |
| `-s` | 1024 | output image size |
| `-bg` | average | background color as hex, e.g. `-bg FFFFFF` |
| `-j` | all cores | number of parallel worker processes |
| `-nth` | 1 | with a `%d` output pattern, save every Nth frame |
| `--candidates` | 200 | random shapes tried at the start of each hill climb |
| `--age` | 100 | failed mutations before a hill climb gives up |
| `--tries` | max(8, `-j`) | hill climbs per shape; the best one wins |
| `--seed` | random | seed for reproducible results |
| `-v` | off | print progress |

### Examples

```bash
# 200 triangles, saved as PNG and SVG
python -m abstractshapes -i photo.jpg -o art.png -o art.svg -n 200 -v

# mix of every shape type, white background
python -m abstractshapes -i photo.jpg -o art.png -n 150 -m 0 -bg FFFFFF

# animated GIF of the drawing process
python -m abstractshapes -i photo.jpg -o anim.gif -n 100 -m 3

# one PNG per shape: frame001.png, frame002.png, …
python -m abstractshapes -i photo.jpg -o frame%03d.png -n 100
```

### Python API

```python
from PIL import Image
from abstractshapes import Model

model = Model(Image.open("photo.jpg").resize((256, 192)), workers=4)
for _ in range(100):
    score = model.step(mode=1, alpha=128)
model.render(size=1024).save("art.png")
open("art.svg", "w").write(model.svg(size=1024))
model.close()
```

## Performance tips

Being pure Python + NumPy, this is slower than the original Go program.
To speed things up:

- lower `-r` (e.g. `-r 128`) — the working resolution matters most;
- use every core with `-j` (the default);
- trade quality for speed with smaller `--candidates`, `--age` or `--tries`.

Output resolution (`-s`) has almost no effect on speed, since shapes are
re-rendered at full size only when saving.

## Credits

This project is a Python recreation of
[**primitive**](https://github.com/fogleman/primitive) — *"Reproducing images
with geometric primitives"* — created by
[Michael Fogleman](https://github.com/fogleman) and released under the MIT
License. The algorithm, the shape modes and their numbering, the command-line
flags and the overall approach all come from his original Go implementation.
All credit for the idea goes to him; any bugs here are our own.

The example painting on the website is *Shakuntala Patralekhan* by
**Raja Ravi Varma** (1848–1906), which is in the public domain, via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Raja_Ravi_Varma_-_Shakuntala_writing_a_love_letter_on_a_lotus_leaf.jpg).

## License

MIT — see [LICENSE](LICENSE).
