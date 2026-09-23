# CLAUDE.md

## Project summary

`photo-to-coloring-page` is a Python CLI that converts photos (JPG/PNG)
into printable black-and-white coloring-page line art. The default
conversion engine (`lineart-raster`) is a pretrained detail-line network,
measured against this project's classical computer-vision engines
(Canny edge detection, adaptive thresholding, an XDoG mode, and others)
with `scripts/ablation.py` -- see the README's "Conversion styles"
section for the numbers. Every classical engine has zero
machine-learning dependencies and remains selectable via `--style`
without installing any extra; `lineart-raster` and every other ML-backed
engine are each gated behind their own optional extra. A pluggable
`ConversionEngine` interface (returning either vector `Drawing` or raster
`RasterArtwork` output -- see `coloring_page.artwork`) allows further
engines to be added without touching the CLI or pipeline.

## Dev environment setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests, lint, and type-check

```bash
pytest                     # run the test suite
pytest --cov=coloring_page --cov-report=term-missing  # with coverage
ruff format .               # format
ruff check .                 # lint
mypy src                    # type-check
```

## Coding conventions

- `src/` layout under `src/coloring_page/`; packaging via `pyproject.toml`
  (PEP 621) only, no `setup.py`.
- Type hints on all public functions; docstrings in NumPy style, explaining
  *why* a parameter or behavior exists, not just restating the signature.
- All comments and docstrings in English.
- No hardcoded paths, secrets, or credentials. Any future configuration
  goes through environment variables documented in `.env.example` — never
  commit a real `.env`.
- New conversion engines/styles go in `src/coloring_page/engines/` as a
  class implementing `ConversionEngine` (see `engines/base.py`), then get
  registered in `engines/registry.py`'s `ENGINES` mapping so they
  automatically appear as a `--style` CLI choice. See
  `.claude/skills/add-conversion-style/SKILL.md` for the full checklist,
  including the requirement to check and document the license of any
  external model before it is used.
- Tests live under `tests/`, mirroring the `src/` package structure. Use
  synthetic images generated in-memory (numpy/PIL) rather than committing
  binary fixtures.

## Commit rules (must follow exactly)

- Plain, conventional-commit style messages (`feat:`, `fix:`, `test:`,
  `docs:`, `chore:`), written as if by an anonymous contributor.
- Do **not** add any AI-attribution trailer or footer to commits — no
  "Generated with ..." lines, no `Co-Authored-By` referencing an AI tool or
  vendor, and no mention of any AI assistant, vendor, or the repository
  owner in commit messages, code comments, docstrings, or any other file.
- Do not push to any remote. No `git remote add`, no `git push`. All
  commits stay local; leave remote configuration to whoever owns the repo.
