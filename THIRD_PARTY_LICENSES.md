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
