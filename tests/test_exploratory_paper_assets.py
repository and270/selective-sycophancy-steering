# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "build_exploratory_paper_assets.py"
SPEC = importlib.util.spec_from_file_location("build_exploratory_paper_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tracked_exploratory_summary_builds_complete_assets() -> None:
    payload = MODULE.load_summary(MODULE.DEFAULT_SOURCE)
    scale_payload = MODULE.load_scale_report(MODULE.DEFAULT_SCALE_SOURCE)
    outputs = MODULE.build_all(payload, scale_payload)

    assert set(outputs) == {
        "model_diagnostics.tex",
        "primary_results.tex",
        "full_frontier_results.tex",
        "intervention_scale_table.tex",
        "frontier_plot.tex",
        "dose_response_plot.tex",
    }
    assert "Qwen3.5-4B" in outputs["primary_results.tex"]
    assert "15.72 [14.51, 16.89]" in outputs["primary_results.tex"]
    assert "21.03 [19.56, 22.47]" in outputs["primary_results.tex"]
    assert "0.106877" in outputs["full_frontier_results.tex"]

    dose = outputs["dose_response_plot.tex"]
    expected_series = {
        "Qwen3.5-2B": ("qwenSmall", "*", "0.37835793"),
        "Qwen3.5-4B": ("qwenLarge", "square*", "15.716096"),
        "Gemma 4 E2B": ("gemmaSmall", "triangle*", "1.0970464"),
        "Gemma 4 E4B": ("gemmaLarge", "diamond*", "21.034601"),
    }
    for label, (color, marker, alpha_two_value) in expected_series.items():
        plot = rf"\addplot[color={color}, mark={marker}, mark options="
        assert plot in dose
        assert rf"\addlegendentry{{{label}}}" in dose
        assert f"(2.0,{alpha_two_value})" in dose

    assert "The red-diamond series is Gemma 4 E4B" in dose
    assert "black!45, thin, forget plot" in dose
    assert "\\addplot+" not in dose

    scale = outputs["intervention_scale_table.tex"]
    assert "Qwen3.5-2B & 2.77\\% & 0.23 & 0.00082" in scale
    assert "Qwen3.5-4B & 14.21\\% & 9.44 & 0.02239" in scale
    assert "Gemma 4 E2B & 15.19\\% & 0.55 & 0.02106" in scale
    assert "Gemma 4 E4B & 11.37\\% & 10.69 & 0.01760" in scale


def test_exploratory_summary_hash_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        MODULE.load_summary(MODULE.DEFAULT_SOURCE, expected_sha256="0" * 64)


def test_intervention_scale_hash_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        MODULE.load_scale_report(MODULE.DEFAULT_SCALE_SOURCE, expected_sha256="0" * 64)
