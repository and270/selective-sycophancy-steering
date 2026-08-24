# Contributing

Contributions are welcome when they preserve the separation between engineering verification and scientific evidence.

## Development setup

```bash
uv sync --group dev --extra quantization --extra benchmarks
uv run --extra quantization --extra benchmarks python -m pytest -q
uv run --extra quantization --extra benchmarks ruff check src tests scripts
uv run --extra quantization --extra benchmarks ruff format --check src tests scripts
uv run --extra quantization --extra benchmarks ty check src
uv run --with bandit bandit -q -r src
```

## Scientific-contract changes

Any change to models, revisions, data IDs/hashes, prompts, estimators, null controls, layer gates, coefficients, decoding, EOS inventories, metrics, or uncertainty must occur before a study freeze or in a new versioned protocol. Never overwrite completed artifacts.

Scientific commands must run through:

```bash
bash scripts/scientific.sh <subcommand> ...
```

The runner rejects dirty or untagged code, contaminated package paths, protocol overrides, parent-artifact drift, and incomplete outputs.

## Tests first

Add a failing regression before changing behavior. Include adversarial tests for fail-closed boundaries. Engineering smokes must use synthetic prompts and must never be cited as model-efficacy evidence.

## Pull requests

Explain:

1. the behavior or contract changed;
2. tests added;
3. whether any preregistration or artifact compatibility changes;
4. model/data/license implications;
5. exact verification commands and outputs.
