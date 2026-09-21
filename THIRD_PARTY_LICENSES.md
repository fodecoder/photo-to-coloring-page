# Third-party licenses

This project vendors a small amount of third-party source code. Each piece
is listed below with its origin and license.

## Anime2Sketch U-Net generator architecture

- **File**: `src/coloring_page/engines/_anime2sketch_arch.py`
- **Source**: https://github.com/Mukosame/Anime2Sketch (`model.py`)
- **License**: MIT
- **Copyright**: (c) 2021 Xiaoyu Xiang

The model definition (the `UnetGenerator`/`UnetSkipConnectionBlock` classes)
is reproduced here, with only cosmetic changes (type hints, docstring
formatting), so that this project's `Anime2SketchEngine` can build a
matching network and load community-provided pretrained weights into it.
No pretrained weights are included in this repository; see the README's
"Extending with an ML engine" section for how to obtain them separately.

Full license text:

```
MIT License

Copyright (c) 2021 Xiaoyu Xiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Informative Drawings generator architecture

- **File**: `src/coloring_page/engines/_informative_drawings_arch.py`
- **Source**: https://github.com/carolineec/informative-drawings (`model.py`,
  the `Generator`/`ResidualBlock` classes only)
- **License**: MIT
- **Copyright**: (c) 2022 Caroline Chan

The generator architecture (a ResNet-style encoder/residual/decoder network,
not the U-Net used by Anime2Sketch) is reproduced here, with only cosmetic
changes (type hints, docstring formatting), so this project's
`InformativeDrawingsEngine` can build a matching network and load
community-provided pretrained weights into it. No pretrained weights are
included in this repository; see the README's "Extending with an ML engine"
section for how to obtain them separately.

**Why weights are downloaded from the official source and not via
`controlnet_aux`/Hugging Face**: `controlnet_aux` (Apache-2.0) offers a
`LineartDetector` that loads these same weights, repackaged on the Hugging
Face Hub as `lllyasviel/Annotators`. That model repository's own declared
license is `other`, with no README and no license text -- it does not state
that it carries forward the upstream MIT terms, and per this project's
`add-conversion-style` skill checklist, "permits the intended use, including
commercial/redistribution use" must be confirmed before merging, not
assumed. Since it could not be confirmed for that specific mirror, this
project instead vendors the architecture directly and documents downloading
weights from the original, unambiguously MIT-licensed
`carolineec/informative-drawings` repository (see the README).

### Update: `lllyasviel/Annotators` now used as an automation source (accepted risk)

The paragraph above documents this project's original decision to reject
the Hugging Face mirror `lllyasviel/Annotators` as a weights source,
because its declared license (`other`) doesn't confirm it carries forward
`informative-drawings`'s upstream MIT terms. That rationale is left
unchanged above -- it accurately describes why the mirror wasn't used at
the time.

This project's owner has since decided, explicitly and knowingly, to use
that same mirror anyway, purely for automation convenience: the official
Google Drive source cannot be scripted (no direct download link, requires
manual interaction), and a one-command/CI-friendly setup needs something
scriptable. `scripts/fetch_weights.py` downloads from
`lllyasviel/Annotators` by default, and `informative_drawings.py`'s
missing-weights error now leads with that script.

**This is an accepted risk, not a resolved question**: the mirror's
redistribution terms are still unconfirmed to carry forward MIT. The
official `carolineec/informative-drawings` Google Drive source remains
documented in the README as the preferred, unambiguously-licensed
alternative for anyone who wants to avoid this risk -- `fetch_weights.py`
is offered alongside it, not instead of it. This entry supersedes only the
"how weights are obtained by default" decision above, not the license
analysis itself.

Full license text:

```
MIT License

Copyright (c) 2022 Caroline Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SAM 2 (`lineart` engine, Stage A)

- **Package**: `sam2` (the `lineart` optional extra)
- **Source**: https://github.com/facebookresearch/sam2
- **License**: Apache License 2.0
- **Copyright**: (c) Meta Platforms, Inc. and affiliates
- **Weights**: `sam2.1_hiera_small.pt`, from the official
  `facebook/sam2.1-hiera-small` Hugging Face repository (no mirror --
  unlike Informative Drawings, this is the project's own official
  redistribution). SAM 2's checkpoints are released under the same
  Apache-2.0 terms as the code.

Nothing from `sam2` is vendored into this repository's source -- it is
used only as an installed dependency
(`src/coloring_page/engines/_lineart_seg.py`), so no license text
reproduction is required here beyond attribution; the full Apache-2.0
text is available at
https://github.com/facebookresearch/sam2/blob/main/LICENSE.

## controlnet_aux `lineart_anime` checkpoint (`lineart` engine, Stage B) -- UNVERIFIED, BLOCKS MERGE

- **Package**: `controlnet_aux` (the `lineart` optional extra)
- **Checkpoint**: `netG.pth`, served from the `lllyasviel/Annotators`
  Hugging Face mirror (see `scripts/fetch_weights.py`) -- the same
  repository already flagged above as an accepted, still-unconfirmed
  redistribution-license risk for the Informative Drawings checkpoints.
- **`controlnet_aux` package license**: Apache License 2.0 (confirmed).
- **Checkpoint's own upstream license**: **not yet confirmed** as of this
  engine's implementation. `lineart_anime` traces back to a distinct
  upstream project from Informative Drawings/Anime2Sketch (both already
  documented above); that upstream project's own license, and whether
  `lllyasviel/Annotators`'s redistribution of it carries those terms
  forward, has not been identified and verified here.

**This project's `add-conversion-style` skill requires the license of any
external model to be identified and confirmed to permit the intended use
(including commercial/redistribution use) before a new ML-backed engine is
merged.** That step is not complete for this checkpoint -- `lineart.py`'s
Stage B is implemented and tested (with injected stubs, no real weights
involved), but this checkpoint must not be fetched or relied on for real
output, and this engine must not be promoted past `experimental` status,
until this entry is replaced with a confirmed upstream license (or an
explicit, documented accepted-risk decision, mirroring the Informative
Drawings precedent above).
