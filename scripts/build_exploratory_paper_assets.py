# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build manuscript tables and plots from the endpoint-verified summary.

The 3.7 GB raw run directories are intentionally excluded from Git.  The
tracked summary used here was independently recomputed from those artifacts;
its pinned SHA-256 prevents a later edit from silently changing the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY / "results" / "FOUR_MODEL_EXPLORATORY_FRONTIER.json"
DEFAULT_SCALE_SOURCE = REPOSITORY / "results" / "INTERVENTION_SCALE_COMPARISON.json"
DEFAULT_OUTPUT = REPOSITORY / "paper" / "generated"
EXPECTED_SOURCE_SHA256 = (
    "cf3ff0773b0df82418200576a8c7e09825b82306bfa16c4ac09fb180ab07f421"
)
EXPECTED_SCALE_SOURCE_SHA256 = (
    "652dc2eeba7d26fa457f51227bd930a50a874add4909ec4018e312ab971af518"
)

MODEL_ORDER = (
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Gemma 4 E2B",
    "Gemma 4 E4B",
)

PLOT_STYLES = {
    "Qwen3.5-2B": ("qwenSmall", "*"),
    "Qwen3.5-4B": ("qwenLarge", "square*"),
    "Gemma 4 E2B": ("gemmaSmall", "triangle*"),
    "Gemma 4 E4B": ("gemmaLarge", "diamond*"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(
    path: Path, *, expected_sha256: str = EXPECTED_SOURCE_SHA256
) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            "Exploratory summary SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = "four_model_frontier_verification.v2"
    if payload.get("schema_version") != expected_schema:
        raise ValueError("Unexpected exploratory-summary schema")
    evidence_scope = payload.get("evidence_scope")
    if not isinstance(evidence_scope, dict):
        raise ValueError("The tracked report is missing its evidence scope")
    if evidence_scope.get("endpoint_results") != (
        "verified_from_persisted_response_and_metric_primitives"
    ):
        raise ValueError("The report must identify its verified endpoint evidence")
    if evidence_scope.get("layer_selection") != (
        "held_out_probe_with_five_seeded_random_controls_per_layer"
    ):
        raise ValueError("The report must identify its executed layer screen")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_ORDER):
        raise ValueError("Exploratory summary has an unexpected model panel")
    for label in MODEL_ORDER:
        model = models[label]
        trials = model.get("trials")
        if not isinstance(trials, list) or [trial.get("alpha") for trial in trials] != [
            -0.5,
            -1.0,
            -2.0,
        ]:
            raise ValueError(f"{label} does not contain the complete alpha frontier")
        if model.get("random_direction_controls") != 5:
            raise ValueError("Exploratory random-control count drifted")
        for trial in trials:
            for key in (
                "natural_stubbornness_95_ci_pp",
                "controlled_stubbornness_95_ci_pp",
            ):
                interval = trial.get(key)
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(f"{label} is missing paired correction intervals")
    return payload


def load_scale_report(
    path: Path, *, expected_sha256: str = EXPECTED_SCALE_SOURCE_SHA256
) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            "Intervention-scale report SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "intervention_scale_comparison.v1":
        raise ValueError("Unexpected intervention-scale report schema")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Intervention-scale report is missing its model panel")
    labels = [model.get("model_label") for model in models if isinstance(model, dict)]
    if labels != list(MODEL_ORDER):
        raise ValueError("Intervention-scale model order differs from the paper panel")
    return payload


def _trial(model: dict[str, Any], alpha: float) -> dict[str, Any]:
    matches = [item for item in model["trials"] if float(item["alpha"]) == alpha]
    if len(matches) != 1:
        raise ValueError(f"Expected one alpha={alpha:g} trial")
    return matches[0]


def _pp(value: float) -> str:
    return f"{float(value):.2f}"


def _ci(values: list[float]) -> str:
    return f"[{float(values[0]):.2f}, {float(values[1]):.2f}]"


def _percent_count(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({100.0 * count / denominator:.2f}\\%)"


def build_model_table(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for label in MODEL_ORDER:
        model = payload["models"][label]
        base = model["base"]
        rows.append(
            " & ".join(
                [
                    label,
                    str(model["primary_zero_based_layer"]),
                    f"{float(model['held_out_probe_auroc']):.3f}",
                    f"{float(model['direction_norm']):.4g}",
                    _percent_count(
                        int(base["pressure_error_count"]),
                        int(base["pressure_denominator"]),
                    ),
                    f"{int(base['gsm_correct_count'])}/256",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            "% Generated from the hash-pinned exploratory summary; do not edit.",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            (
                r"\caption{Model-level diagnostics and unsteered baselines. "
                r"Layer indices are zero-based. Direction norms are not "
                r"normalized and therefore are not directly comparable "
                r"across models.}"
            ),
            r"\label{tab:model-diagnostics}",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            (
                r"Model & Layer & Probe AUROC & $\lVert d\rVert_2$ & "
                r"Pressure error & GSM8K \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_primary_table(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for label in MODEL_ORDER:
        model = payload["models"][label]
        trial = _trial(model, -2.0)
        gsm = trial["gsm"]
        rows.append(
            " & ".join(
                [
                    label,
                    (
                        f"{_pp(trial['pressure_reduction_pp'])} "
                        f"{_ci(trial['pressure_reduction_95_ci_pp'])}"
                    ),
                    (
                        f"{_pp(trial['natural_stubbornness_increase_pp'])} "
                        f"{_ci(trial['natural_stubbornness_95_ci_pp'])}"
                    ),
                    (
                        f"{_pp(trial['controlled_stubbornness_increase_pp'])} "
                        f"{_ci(trial['controlled_stubbornness_95_ci_pp'])}"
                    ),
                    f"{float(gsm['delta_pp']):+.2f} {_ci(gsm['paired_95_ci_pp'])}",
                    f"{float(trial['kl']['forward_kl_nats']):.5f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            "% Generated from the hash-pinned exploratory summary; do not edit.",
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            (
                r"\caption{Primary descriptive comparison at $\alpha=-2$. "
                r"Pressure reduction is base minus steered error; positive "
                r"stubbornness is a loss of correct-evidence acceptance. "
                r"Brackets are paired 95\% bootstrap intervals in percentage "
                r"points. KL is token-micro "
                r"$D_{\mathrm{KL}}(p_{\mathrm{base}}\Vert "
                r"p_{\mathrm{steered}})$ in nats.}"
            ),
            r"\label{tab:primary-results}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            (
                r"Model & Pressure reduction & Natural stub. & Controlled stub. & "
                r"GSM8K $\Delta$ & KL \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )


def build_full_table(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for label in MODEL_ORDER:
        model = payload["models"][label]
        for trial in model["trials"]:
            gsm = trial["gsm"]
            rows.append(
                " & ".join(
                    [
                        label,
                        f"{float(trial['alpha']):g}",
                        (
                            f"{_pp(trial['pressure_reduction_pp'])} "
                            f"{_ci(trial['pressure_reduction_95_ci_pp'])}"
                        ),
                        (
                            f"{_pp(trial['natural_stubbornness_increase_pp'])} "
                            f"{_ci(trial['natural_stubbornness_95_ci_pp'])}"
                        ),
                        (
                            f"{_pp(trial['controlled_stubbornness_increase_pp'])} "
                            f"{_ci(trial['controlled_stubbornness_95_ci_pp'])}"
                        ),
                        f"{float(gsm['delta_pp']):+.2f} {_ci(gsm['paired_95_ci_pp'])}",
                        f"{float(gsm['exact_two_sided_sign_p']):.3f}",
                        f"{float(trial['kl']['forward_kl_nats']):.6f}",
                        f"{100.0 * float(trial['kl']['top1_agreement']):.2f}",
                    ]
                )
                + r" \\"
            )
    return "\n".join(
        [
            "% Generated from the hash-pinned exploratory summary; do not edit.",
            r"\begin{table}[p]",
            r"\centering",
            r"\scriptsize",
            (
                r"\caption{Complete executed frontier. All changes are "
                r"percentage points. Pressure and correction intervals use "
                r"paired source-stratified record-cluster bootstraps; GSM8K "
                r"intervals are paired item bootstraps; "
                r"$p_{\mathrm{sign}}$ is the exact two-sided sign test on "
                r"discordant pairs. Top-1 is token-level agreement on the "
                r"frozen neutral trajectories.}"
            ),
            r"\label{tab:full-frontier}",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{@{}lrrrrrrrr@{}}",
            r"\toprule",
            (
                r"Model & $\alpha$ & Pressure reduction & Natural stub. & "
                r"Controlled stub. & GSM8K $\Delta$ & $p_{\mathrm{sign}}$ & "
                r"KL & Top-1 (\%) \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_scale_table(payload: dict[str, Any], scale_payload: dict[str, Any]) -> str:
    scale_by_label = {model["model_label"]: model for model in scale_payload["models"]}
    rows: list[str] = []
    for label in MODEL_ORDER:
        trial = _trial(payload["models"][label], -1.0)
        updates = scale_by_label[label]["relative_update_by_alpha_magnitude"]
        matched = [
            update for update in updates if float(update["alpha_magnitude"]) == 1.0
        ]
        if len(matched) != 1:
            raise ValueError(f"Expected one |alpha|=1 scale entry for {label}")
        rows.append(
            " & ".join(
                [
                    label,
                    f"{float(matched[0]['percent_of_mean_probe_residual_norm']):.2f}\\%",
                    f"{float(trial['pressure_reduction_pp']):.2f}",
                    f"{float(trial['kl']['forward_kl_nats']):.5f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            "% Generated from hash-pinned endpoint and intervention-scale "
            "reports; do not edit.",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            (
                r"\caption{Residual-relative magnitude at $\alpha=-1$. "
                r"Here $\rho=\lVert\alpha d\rVert_2/"
                r"\mathbb{E}_{i\in\mathrm{probe}}\lVert h_i\rVert_2$ at the "
                r"selected layer. This re-expression changes the unit of dose, "
                r"not the executed intervention.}"
            ),
            r"\label{tab:relative-scale}",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            (
                r"Model & Relative update $\rho$ & Pressure reduction (pp) "
                r"& Forward KL \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _coordinates(model: dict[str, Any], x_key: str, y_key: str) -> str:
    points = []
    for trial in model["trials"]:
        x_source: Any = trial
        for part in x_key.split("."):
            x_source = x_source[part]
        y_source: Any = trial
        for part in y_key.split("."):
            y_source = y_source[part]
        points.append(f"({float(x_source):.8g},{float(y_source):.8g})")
    return " ".join(points)


def build_frontier_plot(payload: dict[str, Any]) -> str:
    first_panel: list[str] = []
    second_panel: list[str] = []
    for label in MODEL_ORDER:
        model = payload["models"][label]
        color, mark = PLOT_STYLES[label]
        common = (
            f"color={color}, mark={mark}, "
            f"mark options={{draw={color}, fill={color}}}, "
            "very thick, mark size=2.4pt"
        )
        frontier_coordinates = _coordinates(
            model,
            "pressure_reduction_pp",
            "natural_stubbornness_increase_pp",
        )
        kl_coordinates = _coordinates(
            model,
            "kl.forward_kl_nats",
            "pressure_reduction_pp",
        )
        first_panel.extend(
            [
                (
                    rf"\addplot[{common}] coordinates "
                    rf"{{{frontier_coordinates}}};"
                ),
                rf"\addlegendentry{{{label}}}",
            ]
        )
        for trial in model["trials"]:
            x_value = float(trial["pressure_reduction_pp"])
            lower, upper = trial["natural_stubbornness_95_ci_pp"]
            first_panel.extend(
                [
                    (
                        rf"\draw[color={color}, opacity=0.65, line width=0.6pt] "
                        rf"(axis cs:{x_value:.8g},{float(lower):.8g}) -- "
                        rf"(axis cs:{x_value:.8g},{float(upper):.8g});"
                    ),
                    (
                        rf"\draw[color={color}, opacity=0.65, line width=0.6pt] "
                        rf"([xshift=-1.5pt]axis cs:{x_value:.8g},{float(lower):.8g}) "
                        rf"-- ([xshift=1.5pt]axis cs:{x_value:.8g},{float(lower):.8g});"
                    ),
                    (
                        rf"\draw[color={color}, opacity=0.65, line width=0.6pt] "
                        rf"([xshift=-1.5pt]axis cs:{x_value:.8g},{float(upper):.8g}) "
                        rf"-- ([xshift=1.5pt]axis cs:{x_value:.8g},{float(upper):.8g});"
                    ),
                ]
            )
        second_panel.extend(
            [
                (
                    rf"\addplot[{common}] coordinates "
                    rf"{{{kl_coordinates}}};"
                ),
                rf"\addlegendentry{{{label}}}",
            ]
        )
    return "\n".join(
        [
            "% Generated from the hash-pinned exploratory summary; do not edit.",
            r"\begin{figure}[t]",
            r"\centering",
            r"\begin{tikzpicture}",
            (
                r"\begin{groupplot}[group style={group size=2 by 1, "
                r"horizontal sep=1.25cm}, width=0.47\textwidth, "
                r"height=0.34\textwidth, grid=major, grid style={draw=black!8}, "
                r"axis line style={black!50}, tick label style={font=\scriptsize}, "
                r"label style={font=\small}, legend style={font=\footnotesize, "
                r"draw=black!18, fill=white, fill opacity=0.92, text opacity=1, "
                r"cells={anchor=west}, inner sep=2pt, row sep=-1pt}, "
                r"legend columns=1]"
            ),
            (
                r"\nextgroupplot[xlabel={Pressure-error reduction (pp)}, "
                r"ylabel={Natural stubbornness increase (pp)}, xmin=-0.6, "
                r"xmax=22.5, ymin=-0.35, ymax=7.6, legend pos=north west]"
            ),
            *first_panel,
            (
                r"\nextgroupplot[xlabel={Forward KL (nats, log scale)}, "
                r"ylabel={Pressure-error reduction (pp)}, xmode=log, "
                r"xmin=0.0005, xmax=0.14, ymin=-0.8, ymax=22.5, "
                r"legend pos=north west]"
            ),
            *second_panel,
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            (
                r"\caption{The executed dose frontier (points progress through "
                r"$\alpha=-0.5,-1,-2$). Left: efficacy against the natural "
                r"correct-evidence cost with paired 95\% bootstrap whiskers; "
                r"points nearer the lower-right are more selective. Right: "
                r"neutral-distribution movement does not by "
                r"itself imply targeted efficacy: Gemma 4 E2B reaches high KL "
                r"with little pressure-error reduction.}"
            ),
            r"\label{fig:frontier}",
            r"\end{figure}",
            "",
        ]
    )


def build_dose_plot(payload: dict[str, Any]) -> str:
    series: list[str] = []
    for label in MODEL_ORDER:
        model = payload["models"][label]
        color, mark = PLOT_STYLES[label]
        coordinates = " ".join(
            f"({abs(float(trial['alpha'])):.1f},"
            f"{float(trial['pressure_reduction_pp']):.8g})"
            for trial in model["trials"]
        )
        series.extend(
            [
                (
                    rf"\addplot[color={color}, mark={mark}, "
                    rf"mark options={{draw={color}, fill={color}}}, very thick, "
                    rf"mark size=2.5pt] coordinates {{{coordinates}}};"
                ),
                rf"\addlegendentry{{{label}}}",
            ]
        )
        for trial in model["trials"]:
            x_value = abs(float(trial["alpha"]))
            lower, upper = trial["pressure_reduction_95_ci_pp"]
            series.extend(
                [
                    (
                        rf"\draw[color={color}, line width=0.7pt] "
                        rf"(axis cs:{x_value:.1f},{float(lower):.8g}) -- "
                        rf"(axis cs:{x_value:.1f},{float(upper):.8g});"
                    ),
                    (
                        rf"\draw[color={color}, line width=0.7pt] "
                        rf"([xshift=-2pt]axis cs:{x_value:.1f},{float(lower):.8g}) "
                        rf"-- ([xshift=2pt]axis cs:{x_value:.1f},{float(lower):.8g});"
                    ),
                    (
                        rf"\draw[color={color}, line width=0.7pt] "
                        rf"([xshift=-2pt]axis cs:{x_value:.1f},{float(upper):.8g}) "
                        rf"-- ([xshift=2pt]axis cs:{x_value:.1f},{float(upper):.8g});"
                    ),
                ]
            )
    return "\n".join(
        [
            "% Generated from the hash-pinned exploratory summary; do not edit.",
            r"\begin{figure}[t]",
            r"\centering",
            r"\begin{tikzpicture}",
            (
                r"\begin{axis}[width=0.78\textwidth, height=0.38\textwidth, "
                r"xlabel={Steering magnitude $|\alpha|$}, "
                r"ylabel={Pressure-error reduction (pp)}, xmin=0.35, xmax=2.15, "
                r"ymin=-1.1, ymax=23.2, xtick={0.5,1,2}, grid=major, "
                r"grid style={draw=black!8}, axis line style={black!50}, "
                r"legend style={font=\footnotesize, draw=black!18, fill=white, "
                r"fill opacity=0.92, text opacity=1, cells={anchor=west}, "
                r"inner sep=2pt, row sep=-1pt}, legend columns=1, "
                r"legend pos=north west]"
            ),
            (
                r"\addplot[black!45, thin, forget plot] coordinates "
                r"{(0.35,0) (2.15,0)};"
            ),
            *series,
            r"\end{axis}",
            r"\end{tikzpicture}",
            (
                r"\caption{Dose response for the primary behavioral endpoint. "
                r"Whiskers are paired 95\% source-stratified record-cluster "
                r"bootstrap intervals. Both 4B arms show a steep response; the "
                r"two smaller arms remain near zero under the executed protocol. "
                r"The red-diamond series is Gemma 4 E4B; the orange-triangle "
                r"series is Gemma 4 E2B.}"
            ),
            r"\label{fig:dose-response}",
            r"\end{figure}",
            "",
        ]
    )


def build_all(payload: dict[str, Any], scale_payload: dict[str, Any]) -> dict[str, str]:
    return {
        "model_diagnostics.tex": build_model_table(payload),
        "primary_results.tex": build_primary_table(payload),
        "full_frontier_results.tex": build_full_table(payload),
        "intervention_scale_table.tex": build_scale_table(payload, scale_payload),
        "frontier_plot.tex": build_frontier_plot(payload),
        "dose_response_plot.tex": build_dose_plot(payload),
    }


def write_outputs(outputs: dict[str, str], output_dir: Path, *, check: bool) -> None:
    mismatches: list[str] = []
    for filename, content in outputs.items():
        path = output_dir / filename
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                mismatches.append(filename)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    if mismatches:
        raise SystemExit("Generated paper assets are stale: " + ", ".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--scale-source", type=Path, default=DEFAULT_SCALE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = load_summary(args.source.resolve())
    scale_payload = load_scale_report(args.scale_source.resolve())
    write_outputs(
        build_all(payload, scale_payload),
        args.output_dir.resolve(),
        check=args.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
