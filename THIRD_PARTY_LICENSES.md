# Third-party licenses

This repository is licensed as a whole under GPL-3.0 (see [`LICENSE`](LICENSE)).
Both sub-projects were originally adapted from MIT-licensed codebases. MIT
permits redistributing modified versions of the code under a different
(including more restrictive) license, provided the original copyright and
permission notice is preserved — reproduced below for each source.

---

## `jepa_simclr_dot_experiments/`

Adapted from the SimCLR-predictor JEPA implementation released alongside:

> V. Sobal, J. S V, S. Jalagam, N. Carion, K. Cho, and Y. LeCun. "Joint
> Embedding Predictive Architectures Focus on Slow Features." SSL Theory and
> Practice Workshop, NeurIPS 2022. [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)

Original license:

```
MIT License

Copyright (c) 2022 Vlad Sobal

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

Two files within Sobal's repo are themselves adapted from Facebook/Meta's
MIT-licensed self-supervised-learning codebases, and that provenance carries
through into this repo unchanged:

- **`resnet.py`** carries a Facebook copyright header pointing at "the LICENSE
  file in the root directory of this source tree" — Facebook's own repo
  (facebookresearch/vicreg), which is MIT-licensed. (Sobal's repo reproduces
  this same file, with the same header pointing outside itself, under its own
  MIT root LICENSE — the mismatch already existed upstream.)
- **`lars.py`** (the `LARS` optimizer and `exclude_bias_and_norm`/
  `adjust_learning_rate` helpers) is adapted from the same
  facebookresearch/vicreg (and facebookresearch/barlowtwins) training code.
  Sobal's repo reproduces it with no header at all; a Facebook MIT header has
  been added to this repo's copy to correct that.

```
Copyright (c) Facebook, Inc. and its affiliates.

MIT License

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

`configs.py`'s `DataclassArgParser` is independently implemented but closely
based on the design of Hugging Face's `HfArgumentParser`
([transformers](https://github.com/huggingface/transformers), Apache License
2.0) — noted here as a design acknowledgment rather than a verbatim-copy
notice, since no code was copied.

---

## `dvsib_pendulum_experiments/`

Adapted from the DySIB (Dynamical Symmetric Information Bottleneck)
implementation released alongside:

> K. M. Martini, E. Abdelaleem, P. Gulati, and I. Nemenman. "Information
> bottleneck for learning the phase space of dynamics from high-dimensional
> experimental data." 2026. [arXiv:2604.24662](https://arxiv.org/abs/2604.24662)

The original DySIB codebase (available at
[github.com/paarthgulati/DYSIB_Pendulum](https://github.com/paarthgulati/DYSIB_Pendulum),
tag `v1.0-arxiv-initial`) was released under MIT by its own authors:

```
MIT License

Copyright (c) 2026 Paarth Gulati and contributors

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
