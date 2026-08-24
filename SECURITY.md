# Security policy

## Reporting

Report vulnerabilities privately to André de Souza Loureiro at `andresloureiross@gmail.com` before public disclosure. Do not include model-provider tokens, private dataset contents, or other third-party credentials in the report. Include a minimal reproduction, affected commit, and impact.

## Scientific supply-chain rules

- Model and dataset revisions are immutable SHA-pinned; every cached checkpoint/config/tokenizer file is content-hashed and the resulting tree must match the preregistered expected hash before deserialization.
- Scientific study, data-lock, and dependency-lock files must be tracked and byte-equivalent to their frozen Git-tag blobs.
- Scientific execution uses the committed `uv.lock` through `scripts/scientific.sh`.
- Git and `nvidia-smi` resolve to absolute non-repository paths whose executable bytes are hashed in provenance.
- Core package origins must resolve under the project `.venv`.
- `PYTHONPATH` contamination is rejected.
- Network-fetched pickle or other executable serialization is forbidden.
- Scientific stages do not load arbitrary remote code.
- Parent artifacts require a status-to-content-manifest hash chain, persisted primitive tensors/responses, recomputed semantic contracts, and the same frozen code identity.
- The GSM8K scorer vendors only the hash-verified lm-eval 0.4.12 task YAML and exact scalar filter/metric semantics; the vulnerable `sqlitedict` dependency is not installed.
- Model weights, tokens, and credentials are never committed.

## Supported version

Security fixes target the current `main` branch and latest tagged release.
