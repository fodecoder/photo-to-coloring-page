---
name: add-conversion-style
description: Guides adding a new photo-to-line-art conversion style/engine to this project, including where to add the engine class, how to register it in the CLI, required tests, and the license-check requirement for any external model.
---

# Adding a new conversion style

Use this checklist whenever asked to add a new coloring-page conversion
style or engine (classical CV or ML-based) to this project.

## 1. Implement the engine

- Add a new module under `src/coloring_page/engines/`, e.g.
  `src/coloring_page/engines/<style_name>.py`.
- Define a class that subclasses `ConversionEngine` from
  `src/coloring_page/engines/base.py` and implements `convert()`.
- Follow the contract documented on `ConversionEngine.convert`: input is a
  BGR `uint8` image of shape `(H, W, 3)`; output must be a single-channel
  `uint8` image of shape `(H, W)` where `255` is background (white paper)
  and lower values are line art.
- Respect the `line_thickness` parameter so the new style stays comparable
  to existing ones.
- Add type hints on all public functions/methods and NumPy-style
  docstrings explaining *why*, not just restating the signature.
- If the new engine has zero extra dependencies (i.e. it's another
  classical CV technique using OpenCV/numpy, already core dependencies),
  it can be a required dependency like the existing engines. If it needs
  extra packages (e.g. a deep-learning framework), add them to the `ml`
  optional dependency group in `pyproject.toml` instead of the core
  `dependencies` list, so the default install stays lightweight.

## 2. Register it

- Add the new class to the `ENGINES` mapping in
  `src/coloring_page/engines/registry.py`. The CLI's `--style` choices are
  derived from this mapping automatically — no changes to `cli.py` are
  needed.

## 3. Write tests

Add a test module under `tests/engines/test_<style_name>.py`, mirroring
the existing engine tests, covering at minimum:

- Output shape matches the input's `(H, W)` and dtype is `uint8`.
- Output looks like line art (e.g. background pixels dominate, or values
  are binary/near-binary, whichever is appropriate for the technique).
- Any style-specific constructor parameters behave as documented.

Prefer synthetic images generated in-memory (numpy/PIL) via the
`synthetic_photo` / `synthetic_photo_path` fixtures in `tests/conftest.py`
rather than committing binary fixtures.

Run the full suite before considering the work done:

```bash
pytest
ruff check .
mypy src
```

## 4. License check (required for any external model)

If the new engine loads a pretrained model or weights from an external
source (e.g. Hugging Face, a research repo, a vendor API):

- Identify and read that model's license (and any dataset license that
  applies to its training data, if relevant).
- Confirm the license permits the intended use (including, if applicable,
  commercial/redistribution use of this project).
- Document the model's name, source, and license in the README's
  "Extending with an ML engine" section (or a dedicated section if the
  project grows more than one ML engine) before merging.
- Never commit model weights to the repository. Document how a user
  downloads/installs them themselves.

Do not merge a new ML-backed engine without completing this step.

## 5. Update docs

- Add the new style to the "Conversion styles" section of `README.md`
  with a short description of how it works and when to prefer it.
