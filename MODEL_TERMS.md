# Model terms and precision policy

Verified against live Hugging Face metadata on 2026-08-05. Immutable model
revisions are pinned across `configs/studies/multimodel_v1.json` and
`configs/studies/multimodel_v1_qwen2b_extension.json`; license terms remain
governed by the upstream repositories.

| Checkpoint | Revision | Hub license tag | Gated | Study precision |
|---|---|---|---:|---|
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Apache-2.0 | no | bf16 |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 | no | bf16 |
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | Apache-2.0 | no | bf16 |
| `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | Apache-2.0 | no | NF4 weights, bf16 compute |

Generation uses frozen EOS inventories rather than tokenizer-only EOS overrides: Qwen `[248044, 248046]`; Gemma E2B/E4B `[1, 106, 50]`.

Weights are never committed or redistributed by this repository. Users download them directly from the upstream Hub repositories.

A steering direction is model-specific and revision-specific. The software does
not treat a vector as portable across model IDs, revisions, tokenizers, chat
templates, hidden widths, block sites, or precision policies. The strict study
pipeline binds and checks those identities automatically, and the tracked
exploratory endpoint runners validate the fit model/study/data-lock and
direction-file hashes. The intentionally low-level `steer_model` API accepts a
caller-owned tensor, so API users remain responsible for matching that tensor
to the checkpoint, layer, representation site, and precision policy.

E4B's bf16 checkpoint exceeds the available 12GB GPU memory. Its NF4 condition is a separately disclosed numerical stratum, not a silent fallback. Base and steered E4B outcomes use the identical quantization policy; cross-model absolute comparisons remain secondary.
