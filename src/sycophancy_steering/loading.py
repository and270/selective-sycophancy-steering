# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pinned Hugging Face loading policies for the preregistered model arms."""

from __future__ import annotations

import gc
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import resolve_transformer_layers


@dataclass
class LoadedStudyModel:
    key: str
    spec: dict[str, Any]
    model: Any
    tokenizer: Any
    text_model: nn.Module
    layers: tuple[nn.Module, ...]
    layer_path: str
    model_class: str
    model_fingerprint: dict[str, Any]
    tokenizer_fingerprint: dict[str, Any]
    generation_eos_token_ids: list[int]
    device: str


def build_quantization_config(model_spec: dict[str, Any]) -> Any | None:
    """Build the exact optional bitsandbytes policy from a model spec."""

    policy = model_spec.get("quantization")
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy.get("method") != "bitsandbytes":
        raise ValueError("Only the frozen bitsandbytes policy is supported")
    from transformers import BitsAndBytesConfig

    dtype_name = policy.get("bnb_4bit_compute_dtype")
    if dtype_name != "bfloat16":
        raise ValueError("Only bfloat16 4-bit compute is supported")
    return BitsAndBytesConfig(
        load_in_4bit=policy.get("load_in_4bit") is True,
        bnb_4bit_quant_type=str(policy.get("bnb_4bit_quant_type")),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=(policy.get("bnb_4bit_use_double_quant") is True),
    )


def configure_torch_runtime(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Runtime seed must be a nonnegative integer")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_content_fingerprint(snapshot: Path) -> dict[str, Any]:
    root = snapshot.resolve()
    if not root.is_dir():
        raise ValueError("Model snapshot path is not a directory")
    paths = sorted(
        (path for path in snapshot.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot).as_posix(),
    )
    if not paths:
        raise ValueError("Model snapshot contains no files")
    files = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    return {
        "snapshot_revision": root.name,
        "file_count": len(files),
        "files": files,
        "content_tree_sha256": _canonical_json_sha256(files),
    }


def _verify_checkpoint_content(
    model_spec: dict[str, Any], checkpoint_content: dict[str, Any]
) -> None:
    if checkpoint_content.get("file_count") != model_spec.get(
        "expected_checkpoint_file_count"
    ):
        raise RuntimeError("Local checkpoint file inventory differs from frozen study")
    if checkpoint_content.get("content_tree_sha256") != model_spec.get(
        "expected_checkpoint_content_tree_sha256"
    ):
        raise RuntimeError("Local checkpoint content tree differs from frozen study")


def _model_fingerprint(
    model: Any, checkpoint_content: dict[str, Any]
) -> dict[str, Any]:
    config = model.config.to_dict()
    parameter_dtypes = Counter(str(parameter.dtype) for parameter in model.parameters())
    quantized_modules = Counter(
        type(module).__name__
        for module in model.modules()
        if "4bit" in type(module).__name__.lower()
        or "8bit" in type(module).__name__.lower()
    )
    quantization = getattr(model.config, "quantization_config", None)
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    return {
        "checkpoint_content": checkpoint_content,
        "config_sha256": _canonical_json_sha256(config),
        "model_type": str(getattr(model.config, "model_type", "unknown")),
        "attention_implementation": str(
            getattr(model.config, "_attn_implementation", "unspecified")
        ),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "stored_parameter_elements": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "parameter_dtype_counts": dict(sorted(parameter_dtypes.items())),
        "quantized_module_counts": dict(sorted(quantized_modules.items())),
        "resolved_quantization_config": quantization,
    }


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary_sha256": _canonical_json_sha256(tokenizer.get_vocab()),
        "special_tokens_sha256": _canonical_json_sha256(tokenizer.special_tokens_map),
        "chat_template_sha256": _canonical_json_sha256(chat_template),
        "padding_side": str(tokenizer.padding_side),
        "truncation_side": str(tokenizer.truncation_side),
        "vocab_size": len(tokenizer),
    }


def load_study_model(
    key: str,
    model_spec: dict[str, Any],
    *,
    seed: int,
    local_files_only: bool,
) -> LoadedStudyModel:
    """Load one immutable checkpoint and verify its architecture contract."""

    if not torch.cuda.is_available():
        raise RuntimeError("The preregistered study requires CUDA")
    configure_torch_runtime(seed)
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForMultimodalLM,
        AutoTokenizer,
    )

    snapshot = Path(
        snapshot_download(
            repo_id=str(model_spec["id"]),
            revision=str(model_spec["revision"]),
            local_files_only=local_files_only,
        )
    )
    if snapshot.resolve().name != model_spec["revision"]:
        raise RuntimeError("Resolved model snapshot does not match immutable revision")
    checkpoint_content = _snapshot_content_fingerprint(snapshot)
    _verify_checkpoint_content(model_spec, checkpoint_content)

    tokenizer: Any = AutoTokenizer.from_pretrained(  # nosec B615
        snapshot.as_posix(),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = build_quantization_config(model_spec)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "device_map": model_spec["device"],
        "low_cpu_mem_usage": True,
    }
    if quantization is not None:
        kwargs["quantization_config"] = quantization
    # The local snapshot tree is pinned, content-hashed, and cannot resolve a Hub ref.
    model = AutoModelForMultimodalLM.from_pretrained(  # nosec B615
        snapshot.as_posix(), **kwargs
    )
    model.eval()

    resolved = resolve_transformer_layers(
        model, expected_layers=int(model_spec["expected_transformer_layers"])
    )
    if resolved.path != model_spec["expected_layer_path"]:
        raise ValueError(
            f"Resolved layer path {resolved.path} differs from preregistration"
        )
    model_class = type(model).__name__
    if model_class != model_spec["model_class"]:
        raise ValueError(
            f"Loaded model class {model_class} differs from preregistration"
        )
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(text_config.hidden_size)
    if hidden_size != int(model_spec["expected_hidden_size"]):
        raise ValueError(
            f"Loaded hidden size {hidden_size} differs from preregistration"
        )
    generation_eos = [int(value) for value in model_spec["generation_eos_token_ids"]]
    if any(value >= len(tokenizer) for value in generation_eos):
        raise RuntimeError(
            "Frozen generation EOS inventory exceeds tokenizer vocabulary"
        )
    default_eos = model.generation_config.eos_token_id
    default_eos_ids = (
        [int(default_eos)]
        if isinstance(default_eos, int)
        else [int(value) for value in default_eos or []]
    )
    if not set(default_eos_ids) <= set(generation_eos):
        raise RuntimeError("Frozen EOS inventory omits a model generation EOS id")
    return LoadedStudyModel(
        key=key,
        spec=model_spec,
        model=model,
        tokenizer=tokenizer,
        text_model=resolved.text_model,
        layers=resolved.layers,
        layer_path=resolved.path,
        model_class=model_class,
        model_fingerprint=_model_fingerprint(model, checkpoint_content),
        tokenizer_fingerprint=_tokenizer_fingerprint(tokenizer),
        generation_eos_token_ids=generation_eos,
        device=str(model_spec["device"]),
    )


def unload_study_model(loaded: LoadedStudyModel) -> None:
    """Release model references held by a mutable loaded-model handle."""

    loaded.layers = ()
    loaded.text_model = nn.Identity()
    loaded.model = None
    loaded.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
