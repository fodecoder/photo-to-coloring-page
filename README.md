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

### As a library

The CLI is a thin wrapper over one function. Embedding this package in a
service (rather than shelling out to the CLI) means calling it directly:

```python
from pathlib import Path
from coloring_page import convert_image, Profile

result = convert_image(Path("photo.jpg"), profile=Profile(detail="child"))
result.save_svg(Path("out.svg"))  # also: save_pdf, save_png

print(result.report)  # QualityReport: ink coverage, enclosed/leaking regions, ...
print(result.timings)  # per-stage duration in seconds, plus "total"
```

`convert_image` also accepts an already-loaded `np.ndarray` (BGR, `uint8`,
shape `(H, W, 3)`) in place of a `Path` — no file I/O happens in that case,
and no `ConversionEngine` ever touches the filesystem. Every failure this
package raises intentionally derives from `coloring_page.ColoringPageError`
— see `src/coloring_page/exceptions.py` for the full typed hierarchy
(`UnsupportedImageError`, `ImageTooLargeError`, `WeightsMissingError`,
`WeightsChecksumError`, `EngineUnavailableError`, `QualityGateError`).

### From the command line

Convert a single photo:

```bash
coloring-page photo.jpg output.svg --detail child
```

```
Converted photo.jpg -> output.svg
```

`--detail` (`toddler` / `child` / `adult`) is the real product parameter —
see [Profiles and detail levels](#profiles-and-detail-levels) below.
Pixel-based `--thickness` is deprecated (still parses, for old scripts, but
has no effect — physical stroke width now lives on `PageSpec.stroke_width_mm`).

Output format is inferred from the output path's extension (`.svg`, `.pdf`,
`.png`), or set explicitly with `--format`:

```bash
coloring-page photo.jpg output --format pdf
```

Save/load a full configuration as JSON, for repeatable conversions or a
service's per-request config:

```bash
coloring-page photo.jpg output.svg --profile profile.json --style chained
```

`--profile` supplies the base configuration; any other flag passed
alongside it overrides just that field. A `Profile` is produced by
`Profile(...).to_json()` in Python, or hand-written as JSON directly (see
`src/coloring_page/profile.py`).

Batch-convert every image in a directory, in parallel:

```bash
coloring-page ./photos ./coloring_pages --style canny --jobs 4
```

```
Converted photos/dog.jpg -> coloring_pages/dog.png
Converted photos/cat.png -> coloring_pages/cat.png
Converted 2/2 images.
```

`--jobs` defaults to `min(4, cpu_count)`. Styles that hold a model
resident in VRAM (`anime2sketch`, `informative_drawings`, `gated`,
`lineart`) always run one file at a time regardless of `--jobs`, since a
process pool would otherwise load one full copy of the model per worker.
One file's failure never aborts the batch — failures are collected and
reported in a final summary, and the process exits non-zero if any file
failed.

Run `coloring-page --help` for the full flag reference.

## Validating output

Nothing about a raster line-art image on its own guarantees it's usable as
a coloring page: a contour with even a tiny gap in it "leaks," so there's
no enclosed area to color inside. Pass `--strict` to check for this and
related problems (ink coverage out of the printable band, contours that
never close, colorable regions too small for a marker) before writing
output:

```bash
coloring-page photo.jpg output.png --style chained --strict
```

On failure, the CLI prints a `QualityReport` (ink coverage, enclosed vs.
leaking region counts, smallest region area, dangling contour endpoints)
to stderr and exits with status 7; in batch mode, per-file failures don't
stop the batch, but the whole invocation still exits non-zero if any file
failed. This -- not `scripts/metrics.py`'s reference-comparison tooling
described below -- is the project's actual acceptance gate for generated
output. See `coloring_page.validate.QualityReport` for the full set of
measurements and their default thresholds.

### Quality report on the reference images

`scripts/report_quality.py` runs `validate()` over this project's 7
reference photos (`docs/starting-image*.jpeg`) and prints this table
directly, instead of the boundary-F1 comparison the project used before
`QualityReport` existed (see `docs/DIAGNOSIS.md` §4 for why F1-against-a-reference
is a poor *acceptance* criterion):

```bash
python scripts/report_quality.py docs --style chained
```

| style | image | ink_coverage | enclosed | leaking | min_region_mm2 | dangling | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chained | starting-image.jpeg | 0.100 | 368 | 269 | 0.0 | 1208 | no |
| chained | starting-image-2.jpeg | 0.078 | 302 | 148 | 0.0 | 698 | no |
| chained | starting-image-3.jpeg | 0.099 | 282 | 71 | 0.0 | 884 | no |
| chained | starting-image-4.jpeg | 0.154 | 805 | 236 | 0.0 | 1872 | no |
| chained | starting-image-5.jpeg | 0.080 | 331 | 188 | 0.0 | 872 | no |
| chained | starting-image-6.jpeg | 0.103 | 354 | 259 | 0.0 | 1190 | no |
| chained | starting-image-7.jpeg | 0.120 | 552 | 332 | 0.0 | 1834 | no |

None of the 7 reference photos currently pass `--strict` at the default
`Profile()` (`detail="child"`, `page=PageSpec()` -- A4 at 300dpi): every
image has a nonzero `leaking` count, meaning `chained`'s raw traced
geometry still has gaps a marker would leak through before any
detail-level filtering closes them. This is measured, current behavior,
not aspirational -- closing that gap (tighter contour-closing in the
engines, or a dedicated closing pass in `apply_detail`) is open work, not
something this table should paper over. `min_region_mm2` of `0.0` reflects
the same root cause: `validate()`'s region-area measurement only considers
regions the border flood-fill finds *enclosed*, and a leaking contour
produces none.

## Profiles and detail levels

`Profile.detail` (`toddler` / `child` / `adult`) is the product-facing
parameter that replaces the old pixel-based `--thickness`: how much of a
source photo's fine structure survives into colorable line art depends on
who's going to color it, not on what resolution the photo happened to be
shot at. It's implemented as a post-conversion filter
(`coloring_page.profile.apply_detail`) over whatever `Drawing` the chosen
engine produced -- no engine needs to know about `detail` itself.

| level | min region area | min stroke length | simplification | intended effect |
| --- | --- | --- | --- | --- |
| `toddler` | 300 mm² | 6 mm | 1.0 mm | few large, simple shapes |
| `child` (default) | 120 mm² | 3 mm | 0.6 mm | moderate detail |
| `adult` | 25 mm² | 1.5 mm | 0.3 mm | most of the engine's native detail |

**Known limitation**: the region-area filter only has an effect for
engines that populate `Path.source_area` (currently `region` and `gated`
only). For every other engine, `toddler`'s "few large regions" effect
comes entirely from the stroke-length and simplification thresholds, not
a true area filter -- a documented approximation, not a silent gap.

## Errors and exit codes

Every intentional failure is a typed exception deriving from
`coloring_page.ColoringPageError` (see `src/coloring_page/exceptions.py`).
The CLI catches these, prints a one-line message to stderr, and exits with
a dedicated code; pass `--verbose` for the full traceback.

| exception | exit code | meaning |
| --- | --- | --- |
| `UnsupportedImageError` | 2 | bad extension, undecodable file, or invalid in-memory array |
| `ImageTooLargeError` | 3 | input exceeds `Profile.max_pixels` (default ~40 megapixels) |
| `WeightsMissingError` | 4 | an ML engine's checkpoint isn't on disk |
| `WeightsChecksumError` | 5 | a checkpoint's SHA256 doesn't match its pinned value |
| `EngineUnavailableError` | 6 | unknown `--style`, or its extra isn't installed |
| `QualityGateError` | 7 | `--strict` was set and the `QualityReport` failed |
| (any other `ColoringPageError`) | 1 | -- |

## Reproducibility

- `Profile.seed` (default `0`) is propagated to `random`, `numpy`, and (if
  installed) `torch`, including `torch.use_deterministic_algorithms(True)`
  where torch supports it. Full CUDA determinism additionally requires the
  `CUBLAS_WORKSPACE_CONFIG` environment variable to be set by the caller
  before the process starts (torch's own requirement, not something this
  package can set on your behalf).
- Seeding does not cross process boundaries: batch mode's
  `ProcessPoolExecutor` workers each re-seed themselves at the start of
  every file.
- Every model checkpoint's SHA256 is verified every time it's loaded --
  not only when `scripts/fetch_weights.py` first downloads it -- via
  `coloring_page.weights.verify_checksum`. A corrupted or tampered file
  fails loudly (`WeightsChecksumError`) rather than silently producing
  unpredictable output.
- Every dependency, including the optional `ml`/`seg`/`lineart` extras, is
  pinned with both a lower and an upper bound in `pyproject.toml`.
- The weights cache directory is configurable as a whole via
  `COLORING_PAGE_WEIGHTS_DIR` (see `.env.example`); once a checkpoint is
  present there, no network access happens -- offline use is the default
  behavior, not a special mode.

## Conversion styles

Coloring-page-appropriate styles (uniform stroke, closed contours, low
noise): `canny`, `chained`, `skeleton`, and `region`. `cartoon` is usable
for painterly sources but tends to run noisier than those four. `adaptive`
and `xdog` are kept as optional styles but are **not** recommended for
producing a coloring page (see below).

This default was chosen using development-time methodology, not the
project's shipped acceptance gate (see [Validating
output](#validating-output) above for that). `scripts/metrics.py` (moved
out of the installed package -- see `docs/DIAGNOSIS.md` §4 for why
comparing to an artistic reference image is a poor *acceptance* criterion,
even though it remains useful for comparing candidate engines against each
other during development) and `scripts/compare.py` measured `chained`
against this project's 3 reference images (boundary F1, normalized against
a degenerate-baseline floor, and whether ink coverage lands in the 3-7%
band a printable coloring page needs): `chained` is the only
zero-dependency style that lands in that ink band on all 3 references, at
a boundary F1 on par with the best of the others (`canny` matches or
slightly beats it on raw F1 but is outside the ink band on 2 of 3). That's
why `chained` is the default, not `canny` as earlier versions of this
project used — see `cli.py`'s `--style` help and `engines/region.py`'s
docstring (which was originally hypothesized to win this comparison and
didn't) for the numbers.

- **`chained`** (default) — flattens small-scale texture with a rolling
  guidance filter, detects *connected edge chains* (not a pixel mask) with
  `cv2.ximgproc.createEdgeDrawing`, then redraws each chain's smoothed
  geometry at a uniform stroke width. Solves broken/jittery contours at
  the source rather than patching a mask afterwards.
- **`canny`** — OpenCV's Canny edge detector, with thresholds derived
  from the image's own gradient-magnitude distribution, on a lightly
  blurred image. Gives crisp, thin outlines on high-contrast photos
  (objects, faces, buildings); may miss soft or low-contrast edges, and
  tends to run over this project's ink-coverage band more often than
  `chained` on the same sources.
- **`xdog`** *(not recommended for coloring pages)* — an eXtended
  Difference-of-Gaussians filter: subtracts two differently-blurred copies
  of the image and sharpens the result with a tanh-based threshold.
  Produces more consistent, artistic-looking outlines and handles gradual
  shading transitions gracefully, but measures far behind every other
  style here (`f1_normalized` averaging ~0.05 across the 3 reference
  images) — it draws noticeably less ink than the reference line art, not
  a leftover bug (the DoG sign error documented in
  `docs/IMPROVEMENT-PROMPT.md` is already fixed). Kept registered for its
  distinct artistic look, not as a coloring-page candidate.
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
- **`region`** — segments the image into flat color regions (L0 gradient
  minimization, then mean-shift filtering or SEEDS/SLIC superpixels), then
  draws only the boundaries between regions whose average colors actually
  differ, redrawn at a uniform stroke width. A more thorough version of the
  same idea `cartoon` uses (region segmentation instead of intensity
  gradients), adding the texture-flattening step, the inter-region-contrast
  filter, and proper geometry redraw that `cartoon` lacks. **Measured
  result**: on this project's 3 reference images, it beats both `canny` and
  `chained` on only 1 of 3 (the other 2 have a known aspect-ratio mismatch
  against their reference — see `scripts/compare.py`'s `REFERENCE_PAIRS`);
  it does have a real ink-admissibility advantage (in this project's 3-7%
  target band on all 3 images, vs 0 of 3 for `canny`) but does not clear
  the bar to replace the gradient-based engines as a default. Kept
  registered as a selectable style, not part of the recommended set; see
  `engines/region.py`'s docstring for the full measurement.
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
- **`lineart`** *(experimental, optional — requires `sam2`/`controlnet_aux`
  setup, see below)* — partitions the photo into semantic regions with
  SAM 2's automatic mask generator (a global, object-identity-based
  question, unlike every gradient/region-color engine above), adds back
  fine internal detail (facial features, folds, deliberate patterns) with
  a dedicated line-extraction network, prunes both against a
  print-size-relative notion of "small enough to matter, big enough to
  color" expressed in millimeters (`min_region_area_mm2`,
  `min_path_length_mm`, converted to pixels via `print_dpi` -- these are
  constructor parameters, not yet exposed as CLI flags), and smooths the
  result with Chaikin corner-cutting. Not yet measured against this project's
  reference images with real weights (see `docs/PRODUCTION-PROMPTS.md`'s
  Prompt 3 pre-merge gates); its Stage B checkpoint's license is also not
  yet confirmed (see `THIRD_PARTY_LICENSES.md`) -- **do not use for real
  output yet**. See `engines/lineart.py`'s docstring for the full design.

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

### `lineart`

`src/coloring_page/engines/lineart.py` wires in SAM 2's automatic mask
generator (Apache License 2.0, Copyright Meta Platforms, Inc.) and
controlnet_aux's `lineart_anime` preprocessor -- see
`THIRD_PARTY_LICENSES.md`. **No pretrained weights are included, and the
`lineart_anime` checkpoint's own upstream license is not yet confirmed --
do not rely on this engine for real output until that entry in
`THIRD_PARTY_LICENSES.md` is resolved.** To use it (once that's done):

1. Install the `lineart` extra: `pip install -e ".[lineart]"` (adds
   `torch`, `sam2`, `controlnet_aux`, `huggingface_hub`, `scipy`).
2. Fetch both checkpoints:
   `python scripts/fetch_weights.py --filename sam2.1_hiera_small.pt` and
   `python scripts/fetch_weights.py --filename netG.pth --dest weights/netG.pth`.
3. Save them to `weights/sam2.1_hiera_small.pt` and `weights/netG.pth`
   (relative to wherever you run the CLI from), to the equivalent paths
   under `~/.cache/coloring_page/`, or set the
   `COLORING_PAGE_LINEART_SEGMENTATION_WEIGHTS` /
   `COLORING_PAGE_LINEART_DETAIL_WEIGHTS` environment variables.

Once both are in place, `lineart` appears as a `--style` choice with
`--show-experimental`. Without them, it's simply absent, and the default
install stays free of ML dependencies.

### Weights cache directory

Every ML engine resolves its checkpoint in the same order: an explicit
constructor argument > the engine's own deprecated per-file environment
variable (kept only where a single shared directory can't express a
choice, e.g. Informative Drawings' `sk_model.pth` vs. `sk_model2.pth`) >
`COLORING_PAGE_WEIGHTS_DIR/<filename>` > a local `./weights/<filename>` >
the per-user cache directory (`~/.cache/coloring_page/` by default). Set
`COLORING_PAGE_WEIGHTS_DIR` once (see `.env.example`) to point every
engine at one shared, pre-populated directory -- e.g. a read-only volume
mount in a service deployment -- instead of setting one variable per
engine. Once a checkpoint is present at its resolved path, no network
access happens: this is the default behavior, not a separate "offline
mode." Every checkpoint's SHA256 is verified every time it's loaded (see
[Reproducibility](#reproducibility) above), so a corrupted or substituted
file fails loudly instead of silently producing bad output.

See `.claude/skills/add-conversion-style/SKILL.md` for the full checklist
when adding a further new engine.

### Future ML engine candidates

1. **PidiNet** / **TEED** — lightweight CNN edge detectors (available via
   `controlnet_aux`), faster than a full generator and closer to a
   "hand-drawn" line than Canny. Not yet evaluated.
2. **Sketch Simplification** (`bobbens/sketch_simplification`, Simo-Serra
   et al., SIGGRAPH 2016/2018) — evaluated and **rejected**: its license
   is "freely available for free non-commercial use... only," i.e.
   non-commercial-only, unlike every other model this project uses
   (all MIT). Disqualified outright by this project's own license-check
   standard (`add-conversion-style` skill checklist requires confirming
   commercial/redistribution use is permitted), independent of how well
   it would have performed.
3. **AniLines** (`zhenglinpan/AniLines-Anime-Lineart-Extractor`) —
   evaluated and **not adopted**: the *code* is MIT-licensed, but (a) its
   weights' own license/hosting terms weren't separately verified (the
   same category of gap that sank the `sketch_simplification` idea's
   sibling, and that this project treats as disqualifying until
   confirmed — see `THIRD_PARTY_LICENSES.md`), and (b) it's trained on
   anime cel imagery, the same category as `anime2sketch` — not a
   cleanup cascade over another engine's output the way
   `sketch_simplification` would have been, so it doesn't address the
   gap either idea was meant to close. That gap, measured in this
   project's own comparisons (`gated`'s docstring, `postprocess.py`'s
   commit history), is a *semantic-selection* problem -- classical and
   ML engines alike react to photographic texture (water, foliage) the
   reference illustrations omit entirely -- not a stroke-cleanliness
   problem a simplification/cleanup pass would fix. A cascade stage is
   only worth revisiting if a future engine's *precision* problem is
   traced to noisy geometry rather than semantic misselection.

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

## Hardware requirements

The default `chained` style (and every other classical engine) runs
entirely on CPU with no GPU required. The four ML-backed styles
(`anime2sketch`, `informative_drawings`, `gated`, `lineart`) benefit from a
CUDA GPU but fall back to CPU automatically (`Profile.device="auto"`);
`lineart`'s SAM 2 stage in particular is not practically fast enough on
CPU for interactive use.

Inputs are capped at `Profile.max_pixels` (default ~40 megapixels;
`ImageTooLargeError` above that) so a single request can't allocate an
unbounded amount of memory. Peak memory measured via
`scripts/profile_memory.py` against a synthetic 6000x4000px (24 MP) photo:

| style | peak memory |
| --- | --- |
| `chained` | 252.9 MB (tracemalloc, Windows; Python-object allocations only) |
| `canny` | 189.8 MB (tracemalloc, Windows; Python-object allocations only) |

These numbers were captured with `tracemalloc` on Windows, which tracks
only Python-object allocations and excludes OpenCV's/torch's native
buffers -- true peak RSS is higher. On Linux/macOS, re-run
`scripts/profile_memory.py`, which uses `resource.getrusage` there for an
accurate number, and update this table.

## Limitations

- No machine-learning model weights are bundled with this project.
- The `tracemalloc`-based numbers above (captured on Windows) undercount
  true peak memory; re-measure with `resource.getrusage` on POSIX before
  relying on them for capacity planning.
- None of the 7 reference photos currently pass `--strict` at the default
  profile -- see [Quality report on the reference
  images](#quality-report-on-the-reference-images) above.

## Packaging as a standalone executable

See [`packaging/README.md`](packaging/README.md) for instructions on
building a one-file executable with PyInstaller.

## License

MIT. See [`LICENSE`](LICENSE) for the full text. This project also vendors
a small piece of third-party source code (the optional `anime2sketch`
engine's network architecture); see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for its license.
