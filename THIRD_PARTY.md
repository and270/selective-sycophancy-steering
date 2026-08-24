# Third-party software notices

This file complements `NOTICE.md` and `paper/references.bib`. It is not a substitute for the full license texts distributed with dependencies.

## Heretic

Portions of the activation-intervention implementation and research scaffolding derive from Heretic v1.4.0 by Philipp Emanuel Weidmann and contributors:

- repository: https://github.com/p-e-w/heretic
- license: AGPL-3.0-or-later

This repository therefore remains AGPL-3.0-or-later, retains SPDX headers and notices, and identifies material methodological changes: architecture-independent layer discovery, exact-site hook capture, cache-aware position policy, controlled correct-suggestion evaluation, neutral KL, sampled GSM8K, and fail-closed artifact verification.

## Scientific methods

Activation steering and contrastive activation addition are established methods. Bibliographic credit is provided to Turner et al. and Panickssery et al. in `paper/references.bib`. Directional projection is discussed with citation to Arditi et al.

## Python dependencies

Dependency names and immutable versions are resolved in `uv.lock`. Important upstream projects include:

- PyTorch;
- Hugging Face Transformers, Datasets, Accelerate, and Hub;
- bitsandbytes (E4B NF4 loading only);
- NumPy, PyArrow, safetensors, and PyYAML;
- pytest, Ruff, and ty for verification.

Their respective licenses remain applicable. Except for the task metadata identified below, this repository does not vendor their source code.

## lm-evaluation-harness GSM8K task contract

`src/sycophancy_steering/contracts/gsm8k-cot-zeroshot-lm-eval-0.4.12.yaml` is the exact `gsm8k_cot_zeroshot` task metadata from lm-evaluation-harness 0.4.12 (SHA-256 `c506c7f5c19da2817db443f7e6d943421dfc9237510a70991d2667cf8efce1e0`). It is retained so scientific scoring can enforce the published prompt, filters, stop strings, and exact-match normalization without installing lm-eval's vulnerable `sqlitedict` dependency. lm-evaluation-harness is MIT-licensed; the required text is in `licenses/lm-evaluation-harness-MIT.txt`. The scalar scoring implementation in this repository is independently written from that declarative contract and differentially tested against its edge cases.

## Models

Model weights are downloaded separately from Hugging Face and are not distributed in this repository. See `MODEL_TERMS.md`.
