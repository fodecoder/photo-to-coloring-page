# photo-to-coloring-page

Convert a photo (JPG/PNG) into a printable black-and-white coloring page —
clean line art suitable for printing and coloring in.

The default engine is pure classical computer vision (OpenCV): no GPU, no
external model downloads, no machine-learning dependencies. A pluggable
engine interface allows a deep-learning engine to be added later without
changing the CLI or the rest of the pipeline (see [Extending with an ML
engine](#extending-with-an-ml-engine) below).

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

For contributors (adds `pytest`, `ruff`, `mypy`):

```bash
pip install -e ".[dev]"
```

## Usage

Convert a single photo:

```bash
coloring-page photo.jpg output.png --style canny
```

```
Converted photo.jpg -> output.png
```

Choose a different style and line thickness:

```bash
coloring-page photo.jpg output.png --style xdog --thickness 2
```

Resize large photos before conversion (downscale only, never upscale):

```bash
coloring-page photo.jpg output.png --style adaptive --max-dimension 1600
```

Batch-convert every image in a directory:

```bash
coloring-page ./photos ./coloring_pages --style canny
```

```
Converted photos/dog.jpg -> coloring_pages/dog_coloring.jpg
Converted photos/cat.png -> coloring_pages/cat_coloring.png
```

Run `coloring-page --help` for the full flag reference.

## Conversion styles

- **`canny`** (default) — OpenCV's Canny edge detector on a lightly blurred
  image. Gives crisp, thin outlines on high-contrast photos (objects,
  faces, buildings); may miss soft or low-contrast edges.
- **`adaptive`** — median blur followed by adaptive (per-neighborhood)
  thresholding, similar to a pencil-sketch effect. Preserves more shading
  as texture than `canny`, which can suit portraits or busy scenes.
- **`xdog`** — an eXtended Difference-of-Gaussians filter: subtracts two
  differently-blurred copies of the image and sharpens the result with a
  tanh-based threshold. Tends to produce more consistent, artistic-looking
  outlines and handles gradual shading transitions gracefully.
- **`cartoon`** — quantizes the image to a small flat color palette
  (k-means) and draws lines only at the boundaries between quantized
  regions. Unlike the other three styles, it doesn't key off intensity
  gradients at all, so painted/illustrated sources with lots of internal
  texture, shading, or glow effects don't produce stippling noise. Best
  suited to painterly illustrations rather than photographs.
- **`anime2sketch`** *(optional, requires setup — see below)* — a
  pretrained neural network specifically trained to extract clean line art
  from illustration/anime-style artwork. Only appears as a `--style` choice
  once its setup steps have been completed.

All four built-in styles are implemented in `src/coloring_page/engines/`.

## Extending with an ML engine

`src/coloring_page/engines/base.py` defines an abstract `ConversionEngine`
interface. A pretrained deep-learning model can be added as a new engine
implementing that interface and registered in
`src/coloring_page/engines/registry.py`; the CLI and pipeline require no
other changes. This repository does **not** bundle any model weights, and
the default engine has zero machine-learning dependencies.

### `anime2sketch`

`src/coloring_page/engines/anime2sketch.py` wires in
[Anime2Sketch](https://github.com/Mukosame/Anime2Sketch) (MIT License,
Copyright (c) 2021 Xiaoyu Xiang — see `THIRD_PARTY_LICENSES.md`) as an
example of a real pretrained engine. Its network architecture is vendored
into `src/coloring_page/engines/_anime2sketch_arch.py`; **no pretrained
weights are included**. To use it:

1. Install the `ml` extra: `pip install -e ".[ml]"` (adds `torch`).
2. Download the "default" model's weights yourself from the official
   Google Drive link in the
   [Anime2Sketch README](https://github.com/Mukosame/Anime2Sketch#download-pretrained-weights).
3. Save the file to `~/.cache/coloring_page/anime2sketch.pth`, or set the
   `COLORING_PAGE_ANIME2SKETCH_WEIGHTS` environment variable to point at
   wherever you saved it (see `.env.example`).

Once both steps are done, `anime2sketch` appears as a `--style` choice.
Without them, it's simply absent from `--help` and the default install
stays free of ML dependencies. Before using this or any other pretrained
model, confirm its license permits your intended use.

See `.claude/skills/add-conversion-style/SKILL.md` for the full checklist
when adding a further new engine.

## Running the test suite

```bash
pip install -e ".[dev]"
pytest
```

With coverage:

```bash
pytest --cov=coloring_page --cov-report=term-missing
```

Tests generate their own synthetic input images in-memory (via numpy/PIL);
no binary fixtures are committed to the repository.

## Linting and type-checking

```bash
ruff format .
ruff check .
mypy src
```

## Limitations

- The default engine runs entirely on CPU; there is no GPU acceleration.
- No machine-learning model weights are bundled with this project.
- Very large images may be slow to process; use `--max-dimension` to
  downscale before conversion if needed.

## Packaging as a standalone executable

See [`packaging/README.md`](packaging/README.md) for instructions on
building a one-file executable with PyInstaller.

## License

MIT. See [`LICENSE`](LICENSE) for the full text. This project also vendors
a small piece of third-party source code (the optional `anime2sketch`
engine's network architecture); see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for its license.
