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

All three are implemented in `src/coloring_page/engines/`.

## Extending with an ML engine

`src/coloring_page/engines/base.py` defines an abstract `ConversionEngine`
interface. A pretrained deep-learning model (for example, one hosted on
Hugging Face) can be added as a new engine implementing that interface and
registered in `src/coloring_page/engines/registry.py`; the CLI and pipeline
require no other changes.

This repository does **not** bundle any model weights, and the default
engine has zero machine-learning dependencies. To use an ML-based engine, a
user must install its inference library themselves (see the `ml` optional
dependency group in `pyproject.toml`), download the model weights of their
choice, and check that model's license permits their intended use before
enabling it. See `.claude/skills/add-conversion-style/SKILL.md` for the
full checklist when adding a new engine.

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

MIT. See [`LICENSE`](LICENSE) for the full text.
