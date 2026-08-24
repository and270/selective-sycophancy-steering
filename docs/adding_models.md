# Adding another model

There are two distinct extension paths. Keep them separate:

1. **Apply a known direction to an already-loaded model.** This uses the small public runtime API and does not invoke the study machinery.
2. **Add a model to a reproducible study.** This requires a new versioned protocol, data boundaries, direction fitting, probe controls, evaluation, and artifact provenance.

## Runtime-only integration

Install the package from the tagged source release, then load a direction that
was fitted for the exact checkpoint revision. The `v1.0.0` evidence bundle
provides working examples for all four paper models; see
[`study_reproduction.md`](study_reproduction.md).

Load the model and tokenizer using the policy appropriate for that checkpoint, then apply a model-specific direction as a temporary context manager:

```python
from safetensors.torch import load_file
from sycophancy_steering import resolve_steering_target, steer_model

layer_index = 18
direction = load_file("directions.safetensors", device="cpu")[
    "observed_prompt_state"
][layer_index]

target = resolve_steering_target(model, layer_index=layer_index)
print(target.layer_path, target.layer_count)

with steer_model(
    model,
    direction,
    layer_index=layer_index,
    alpha=-1.0,
) as audit:
    generated = model.generate(**inputs)

if audit.calls == 0:
    raise RuntimeError("The selected decoder block was not executed")
```

The API:

- accepts an already-loaded `torch.nn.Module`;
- infers the expected layer count from common Transformers config fields, or accepts `expected_layers=` explicitly;
- resolves common decoder layouts such as `model.layers`, `model.language_model.layers`, and `transformer.h`;
- modifies the final non-padding position during prefill and the current position during cached decoding;
- removes every hook when the context exits, including after an exception;
- never edits or exports model weights.

The caller remains responsible for `model.eval()`, tokenization, device placement, dtype/quantization, generation arguments, EOS behavior, and checking the yielded hook audit.

## Direction compatibility

A direction is not a portable generic embedding. Bind every released vector to at least:

- model repository ID and immutable revision;
- model/config/tokenizer/chat-template fingerprints;
- decoder block path, zero-based layer, hidden width, and capture site;
- dtype and quantization policy used during fitting and evaluation;
- estimator, sign convention, and tensor hash.

Reject a vector when any of those identities differ. Equal nominal `alpha` values are also not physically comparable across models when raw mean-difference norms differ.

## Supporting another architecture

Before adding a new decoder layout:

1. Identify the exact module whose raw output is the residual representation used for both capture and intervention.
2. Add the narrowest unambiguous path to `src/sycophancy_steering/models.py`.
3. Add a synthetic layer-resolution test and a forward-hook test covering prefill, padding, and cached decode.
4. Run an `alpha=0` identity check and require nonzero hook activity.
5. Verify that the block executes exactly once per text-model forward. Architectures with shared/recurrent blocks require a separate intervention policy rather than relaxing this guard.

Do not enable `trust_remote_code` merely to make an unsupported checkpoint load. Review and pin any new executable model code separately.

## Adding a study arm

Do not edit a frozen study JSON in place. Copy it to a new version and record:

1. immutable model revision and checkpoint content-tree inventory;
2. expected loader class, layer path/count, hidden width, EOS inventory, dtype, device, and quantization;
3. prompt/chat-template behavior and batch-size envelope;
4. physically separate fit, probe, and evaluation records;
5. direction estimator and class/cell minimums;
6. random-axis controls, family-wise layer correction, layer ranking, and fixed alpha grid;
7. pressure, natural correction, controlled correction, KL, and capability outcomes;
8. exact seeds, uncertainty procedure, artifact schema, and failure rules;
9. model/data licenses and redistribution policy.

Run a synthetic or tiny engineering smoke first to validate loading and hook
mechanics. For future confirmatory work, freeze the complete protocol and
execute through `scripts/scientific.sh` so repository, tag, data, and dependency
identities are captured from the start.

For a direct reproduction of the completed five-control fitting protocol, use
`sycophancy-steering fit-probe --run-kind executed_reproduction`. This mode
uses every frozen fit/probe record, the recorded batch sizes, and the executed
five-control layer screen without requiring a future confirmatory study tag.
