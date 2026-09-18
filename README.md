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
coloring-page photo.jpg output.png --style chained --max-dimension 1600
```

Batch-convert every image in a directory:

```bash
coloring-page ./photos ./coloring_pages --style canny
```

```
Converted photos/dog.jpg -> coloring_pages/dog_coloring.png
Converted photos/cat.png -> coloring_pages/cat_coloring.png
```

Batch mode always writes PNG, regardless of the input's format: JPEG's
lossy compression produces visible ringing artifacts around line art.

Run `coloring-page --help` for the full flag reference.

## Conversion styles

Coloring-page-appropriate styles (uniform stroke, closed contours, low
noise): `canny`, `xdog`, `chained`, and `skeleton`. `cartoon` is usable
for painterly sources but tends to run noisier than the four above.
`adaptive` is kept as an optional textured-sketch style but is **not**
recommended for producing a coloring page (see below).

- **`canny`** (default) — OpenCV's Canny edge detector, with thresholds
  derived from the image's own gradient-magnitude distribution, on a
  lightly blurred image. Gives crisp, thin outlines on high-contrast
  photos (objects, faces, buildings); may miss soft or low-contrast edges.
- **`xdog`** — an eXtended Difference-of-Gaussians filter: subtracts two
  differently-blurred copies of the image and sharpens the result with a
  tanh-based threshold. Tends to produce more consistent, artistic-looking
  outlines and handles gradual shading transitions gracefully.
- **`chained`** — flattens small-scale texture with a rolling guidance
  filter, detects *connected edge chains* (not a pixel mask) with
  `cv2.ximgproc.createEdgeDrawing`, then redraws each chain's smoothed
  geometry at a uniform stroke width. Solves broken/jittery contours at
  the source rather than patching a mask afterwards; a good default
  candidate for painterly or illustrated sources.
- **`skeleton`** — an alternative route to the same "redraw the geometry"
  idea: L0 gradient minimization flattens texture, Canny finds edges,
  `cv2.ximgproc.thinning` reduces them to a 1px skeleton (with short
  dead-end spurs pruned), and the result is traced and redrawn. Compare
  against `chained` with `scripts/compare.py` on your own images before
  picking one as a default.
- **`cartoon`** — segments the image into flat regions with mean-shift
  filtering and draws lines only at the boundaries between regions.
  Unlike the gradient-based styles, it doesn't key off intensity
  gradients at all, so painted/illustrated sources with lots of internal
  texture, shading, or glow effects don't produce stippling noise, but
  it runs noisier than `chained`/`skeleton` on the same kind of source.
- **`adaptive`** *(textured sketch, not a coloring-page style)* — median
  blur followed by adaptive (per-neighborhood) thresholding, similar to a
  pencil-sketch effect. Preserves shading as texture rather than clean
  outlines, which produces far too much ink coverage and too many
  disconnected fragments to be usable as a coloring page; kept only for
  users who want that textured-sketch look for its own sake.
- **`informative_drawings`** *(optional, requires setup — see below)* — a
  pretrained neural network (Chan, Durand, Isola, CVPR 2022) trained on
  photographs and paintings paired with hand-drawn line art. Recommended
  over `anime2sketch` for photographed/painted illustrations, since it was
  actually trained on that kind of source rather than already-clean
  digital anime line art.
- **`anime2sketch`** *(optional, requires setup — see below)* — a
  pretrained neural network specifically trained to extract clean line art
  from illustration/anime-style artwork. Only appears as a `--style` choice
  once its setup steps have been completed. Applies gamma correction
  before inference (avoids the network collapsing large dark/shadowed
  regions into a solid black blob) and binarizes its output by default
  for crisp, printable ink lines instead of soft pencil shading.
- **`gated`** *(experimental, optional — requires `informative_drawings`'
  setup)* — runs `chained`'s edge-chain detection permissively, then
  discards chains the `informative_drawings` network isn't confident lie
  on real ink, keeping only the survivors. The idea: `chained` already
  finds most real strokes but can't distinguish an object outline from
  photographic texture (water, foliage) by gradient magnitude alone, and
  the network can make exactly that distinction. **Measured result**:
  this does not currently beat plain `chained` on this project's
  reference images — gating raises precision but recall falls off faster
  than precision rises at every threshold tested, so average boundary F1
  peaks at the no-op threshold (nothing gated) and declines from there.
  Kept registered for experimentation, not part of the recommended set;
  see `engines/gated.py`'s docstring for the full measurement and ideas
  for a better gating aggregate.

All built-in styles are implemented in `src/coloring_page/engines/`.

## Extending with an ML engine

`src/coloring_page/engines/base.py` defines an abstract `ConversionEngine`
interface. A pretrained deep-learning model can be added as a new engine
implementing that interface and registered in
`src/coloring_page/engines/registry.py`; the CLI and pipeline require no
other changes. This repository does **not** bundle any model weights, and
the default engine has zero machine-learning dependencies.

### `informative_drawings`

`src/coloring_page/engines/informative_drawings.py` wires in
[Informative Drawings](https://github.com/carolineec/informative-drawings)
(MIT License, Copyright (c) 2022 Caroline Chan — see
`THIRD_PARTY_LICENSES.md`). Its network architecture is vendored into
`src/coloring_page/engines/_informative_drawings_arch.py`; **no pretrained
weights are included**. To use it:

1. Install the `ml` extra: `pip install -e ".[ml]"` (adds `torch` and
   `huggingface_hub`).
2. Get the weights, either way:
   - **Automated (recommended)**: `python scripts/fetch_weights.py`
     downloads both checkpoints from the `lllyasviel/Annotators` Hugging
     Face mirror and saves them under `~/.cache/coloring_page/`. This
     mirror's own declared license is "other" with no license text
     confirming it carries forward `informative-drawings`'s upstream MIT
     terms -- using it is a deliberate, documented accepted risk; see
     `THIRD_PARTY_LICENSES.md` for the full reasoning and the
     unambiguously-licensed alternative below.
   - **Manual (official, unambiguously licensed)**: download `model.zip`
     from the official Google Drive link in the
     [Informative Drawings README](https://github.com/carolineec/informative-drawings#testing),
     unzip it, and locate `checkpoints/contour_style/netG_A_latest.pth`
     (`contour_style` is recommended for photographed/printed
     illustrations; `anime_style` and `opensketch_style` are also
     included in the same archive). Save it to
     `weights/informative_drawings.pth` (relative to wherever you run the
     CLI from), to `~/.cache/coloring_page/informative_drawings.pth`, or
     set the `COLORING_PAGE_INFORMATIVE_DRAWINGS_WEIGHTS` environment
     variable to point at wherever you saved it (see `.env.example`).

`scripts/fetch_weights.py` downloads both `sk_model.pth` (fine detail) and
`sk_model2.pth` (coarse detail); use `scripts/compare.py` to measure which
one performs better on your own images and point
`COLORING_PAGE_INFORMATIVE_DRAWINGS_WEIGHTS` at that one. Measured against
this project's own three reference pairs (before the shared postprocessing
module -- raw thresholded output only), `sk_model.pth` scored marginally
higher on average (boundary F1 ≈0.27 vs. ≈0.26) -- the opposite of what
"coarse detail should suit page photos better" would predict, which is
exactly why this is measured rather than assumed.

Once the weights are in place, `informative_drawings` appears as a
`--style` choice. Without them, it's simply absent from `--help` and the
default install stays free of ML dependencies.

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
3. Save the file to `weights/anime2sketch.pth` (relative to wherever you
   run the CLI from -- matches the upstream project's own convention), to
   `~/.cache/coloring_page/anime2sketch.pth`, or set the
   `COLORING_PAGE_ANIME2SKETCH_WEIGHTS` environment variable to point at
   wherever you saved it (see `.env.example`).

Once both steps are done, `anime2sketch` appears as a `--style` choice.
Without them, it's simply absent from `--help` and the default install
stays free of ML dependencies. Before using this or any other pretrained
model, confirm its license permits your intended use.

See `.claude/skills/add-conversion-style/SKILL.md` for the full checklist
when adding a further new engine.

### Future ML engine candidates

Evaluated but not implemented in this round, in priority order -- each
would need its own license check before merging:

1. **PidiNet** / **TEED** — lightweight CNN edge detectors (available via
   `controlnet_aux`), faster than a full generator and closer to a
   "hand-drawn" line than Canny.
2. **Sketch Simplification** (Simo-Serra et al., SIGGRAPH 2016/2018) —
   not an extractor but a second stage: takes rough, noisy line art and
   produces clean, uniform strokes. Worth trying *in cascade* after
   `chained`/`skeleton` rather than as a standalone style.
3. **AniLines** (`zhenglinpan/AniLines-Anime-Lineart-Extractor`) — more
   recent; license and weight size not yet verified.

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
