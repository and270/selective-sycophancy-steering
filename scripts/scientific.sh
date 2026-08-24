#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset VIRTUAL_ENV

exec uv run --locked --extra quantization --extra benchmarks \
  sycophancy-steering "$@"
