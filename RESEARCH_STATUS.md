# Research status

**Updated:** 2026-08-24
**Current evidence class:** completed empirical intervention study with verified endpoints

## Direct status

The harmonized Qwen3.5/Gemma 4 panel is complete for all four model arms. The tracked endpoint summary was independently recomputed from the persisted behavior, sampled GSM8K, and neutral-trajectory KL primitives.

| Model | Fit/probe | Behavior frontier | Sampled GSM8K | Neutral KL |
|---|---|---|---|---|
| Qwen3.5-2B | complete | complete | complete | complete |
| Qwen3.5-4B | complete | complete | complete | complete |
| Gemma 4 E2B | complete | complete | complete | complete |
| Gemma 4 E4B | complete | complete | complete | complete |

The outcome-first report is [`results/FOUR_MODEL_EXPLORATORY_FRONTIER.md`](results/FOUR_MODEL_EXPLORATORY_FRONTIER.md). Its machine-readable companion is [`results/FOUR_MODEL_EXPLORATORY_FRONTIER.json`](results/FOUR_MODEL_EXPLORATORY_FRONTIER.json). The derived residual-relative comparison is [`results/INTERVENTION_SCALE_COMPARISON.json`](results/INTERVENTION_SCALE_COMPARISON.json).

## Interpretation and remaining scope

The executed study used 300 fit records, a physically separate 100-record probe split, and a 1,310-record behavioral evaluation split. Five seeded random controls per layer make the layer-screen null coarse, but the selected interventions were evaluated on the separate large split. The reported effects are therefore empirical results of the executed protocol; optimal-layer stability and broader scaling generalization remain follow-up questions.

The bundled direction tensors and downstream artifact identities are hash-bound,
and persisted response/metric primitives support recomputation of every reported
endpoint. Original launch labels and incomplete commit identifiers are retained
for transparency; they limit exact refitting from a known clean source revision,
not the measured response-level results.

Residual-relative calibration gives a sharper size interpretation. At
`|alpha|=1`, Gemma E2B received a 15.19% relative update versus E4B's 11.37%
but remained far less responsive, so the Gemma contrast is not a magnitude
artifact. Qwen2B received 2.77% versus Qwen4B's 14.21%, making a matched-dose
Qwen extension the targeted next comparison.

## Public reproducibility state

- The verified compact report, machine-readable summary, study/data contracts, package, tests, and paper-asset generator are tracked.
- [`results/EXPLORATORY_ARTIFACT_MANIFEST.json`](results/EXPLORATORY_ARTIFACT_MANIFEST.json) inventories the portable release subset: behavior responses, paired GSM8K examples, KL primitives, fit summaries, and selected direction tensors for every model.
- `scripts/package_exploratory_artifacts.py` verifies those files against the
  summary and creates a byte-reproducible, uncompressed Zip64 release at
  `dist/selective-sycophancy-exploratory-artifacts.zip`. The current archive is
  32,921,615 bytes with SHA-256
  `9bcfcd41c158e043b2e10bc517b8b753a63a2c0135390a46c842ec058a5be0b2`.
- The approximately 3.7 GB complete local run tree remains excluded from ordinary Git history. In particular, the curated archive does not include the large all-layer observation tensors needed to refit every direction from residuals, and the original fit launches do not establish clean-commit provenance.
- Source-derived SycophancyEval questions are not redistributed because the upstream repository did not expose an explicit data license at protocol freeze. The deterministic materializer and expected hashes are tracked instead.
- Paper assets can be regenerated and checked without loading a model; see [`docs/study_reproduction.md`](docs/study_reproduction.md).
- The visually verified 18-page local release build is available at
  `output/pdf/selective-resistance-under-pressure.pdf` (156,161 bytes; SHA-256
  `f07bb5c596ef5c91ee6177c96ae5c9d7c7b61a26ac9ef0542b42557af08cf830`).

## Publication record

The final preprint is published under DOI
[`10.5281/zenodo.22082268`](https://doi.org/10.5281/zenodo.22082268). The
`v1.0.0` GitHub release binds this source tree to the byte-verified paper PDF
and deterministic evidence ZIP. No model run or paper-generation task is
pending. Any future 100-control or larger-model study is a new, separately
versioned experiment rather than a continuation required to complete this
panel.
