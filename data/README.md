# Data provenance and materialization

The factual items originate from the `answer.jsonl` split released with [Towards Understanding Sycophancy in Language Models](https://github.com/meg-tong/sycophancy-eval), pinned at commit `9a1694221e3639887138f61deae344335eca6752`.

That upstream repository did not expose an explicit software/data license when this study was frozen. Consequently, this repository commits:

- source identity and canonical SHA-256;
- deterministic transformation code;
- exact split counts, file hashes, and ordered-ID hashes;
- schema and validation tests;

but does **not** redistribute the source-derived question records. Users must obtain the source from its authors and run the materializer locally.

Before assigning any split, the lock generator drops all eight records belonging to four exact normalized-question groups and all 36 records participating in 20 near-duplicate edges. Exact groups use NFKC normalization, case folding, and whitespace collapse. Near duplicates use the maximum bidirectional Python `SequenceMatcher` ratio or whitespace-token Jaccard similarity at a frozen threshold of 0.90. The remaining 1,772-record pool is deterministically source-stratified into 150/50 fit/probe records per source and 732 TriviaQA plus 578 TruthfulQA evaluation records; 59 TriviaQA and three TruthfulQA records are reserved. Thus no defined question family can cross fit, probe, or evaluation boundaries, and every public split is A/B-balanced within source.

Generated files live under `data/materialized/multimodel_v1/` and are ignored by Git:

```text
direction_fit.jsonl     300 records
direction_probe.jsonl   100 records
evaluation.jsonl        1310 records
manifest.json
```

Before any scientific stage, the CLI validates every file against `configs/data/multimodel_v1_data_lock.json`. The materializer applies the frozen exact/near-duplicate exclusion-ID inventory, verifies the resulting parent and split hashes, and enforces pairwise ID/exact-question disjointness; a stage may read only the splits listed in the lock's `scientific_access` boundary.

Generate and verify them with:

```bash
git clone https://github.com/meg-tong/sycophancy-eval data/raw/sycophancy-eval
git -C data/raw/sycophancy-eval checkout 9a1694221e3639887138f61deae344335eca6752
sycophancy-steering materialize-data \
  --source data/raw/sycophancy-eval/datasets/answer.jsonl
```

The command refuses to overwrite an existing output, validates the canonical upstream SHA-256, derives 1,816 valid source pairs, applies the frozen 44-record exact/near-duplicate exclusion inventory, verifies the resulting 1,772-record parent hash, verifies every final split byte hash and ordered-ID hash, and confirms ID and exact-normalized-question disjointness.
