# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line interface for reproducible steering studies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__

DEFAULT_STUDY = Path("configs/studies/multimodel_v1.json")
DEFAULT_DATA_LOCK = Path("configs/data/multimodel_v1_data_lock.json")
DEFAULT_DATA_DIR = Path("data/materialized/multimodel_v1")


def _repository_root() -> Path:
    candidates = (Path.cwd().resolve(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / "configs/studies/multimodel_v1.json").is_file():
            return candidate
    raise RuntimeError(
        "This command requires the selective-sycophancy-steering source checkout; "
        "run it from the repository root"
    )


def _checkout_path(
    value: Path | None,
    relative_default: Path,
    *,
    repository: Path | None = None,
) -> Path:
    if value is not None:
        return value.resolve()
    root = _repository_root() if repository is None else repository
    return (root / relative_default).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sycophancy-steering",
        description=(
            "Fit, probe, and evaluate factual-sycophancy activation steering "
            "under hash-bound study contracts."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser(
        "materialize-data",
        help="Recreate the license-safe factual splits from a pinned source file.",
    )
    materialize.add_argument(
        "--source", type=Path, required=True, help="Pinned upstream answer.jsonl."
    )
    materialize.add_argument(
        "--lock",
        type=Path,
        help="Data-lock JSON (defaults to the checkout's multimodel v1 lock).",
    )
    materialize.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (defaults inside the source checkout).",
    )

    validate = commands.add_parser(
        "validate-study", help="Validate a study JSON without loading a model."
    )
    validate.add_argument(
        "--study",
        type=Path,
        help="Study JSON (defaults to multimodel v1 in the source checkout).",
    )
    validate.add_argument(
        "--require-frozen",
        action="store_true",
        help="Reject draft or otherwise non-launchable study contracts.",
    )

    fit = commands.add_parser(
        "fit-probe", help="Fit directions and select layers on the probe split."
    )
    fit.add_argument("--model-key", required=True, help="Model key in the study JSON.")
    fit.add_argument(
        "--study",
        type=Path,
        help="Study JSON (defaults to multimodel v1 in the source checkout).",
    )
    fit.add_argument(
        "--data-dir",
        type=Path,
        help="Verified materialized-data directory.",
    )
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument(
        "--run-kind",
        choices=("scientific", "executed_reproduction", "engineering_smoke"),
        default="scientific",
        help=(
            "scientific requires a frozen tagged protocol; executed_reproduction "
            "refits the completed five-control study on every frozen fit/probe "
            "record; engineering_smoke requires --limit"
        ),
    )
    fit.add_argument("--limit", type=int)
    fit.add_argument("--generation-batch-size", type=int)
    fit.add_argument("--residual-batch-size", type=int)
    fit.add_argument("--allow-network", action="store_true")

    frontier = commands.add_parser(
        "evaluate-frontier", help="Evaluate the frozen behavioral steering frontier."
    )
    frontier.add_argument(
        "--model-key", required=True, help="Model key in the study JSON."
    )
    frontier.add_argument(
        "--study",
        type=Path,
        help="Study JSON (defaults to multimodel v1 in the source checkout).",
    )
    frontier.add_argument(
        "--data-dir",
        type=Path,
        help="Verified materialized-data directory.",
    )
    frontier.add_argument("--fit-probe-dir", type=Path, required=True)
    frontier.add_argument("--output-dir", type=Path, required=True)
    frontier.add_argument("--allow-network", action="store_true")

    kl = commands.add_parser(
        "evaluate-kl", help="Measure neutral fixed-trajectory distribution shift."
    )
    kl.add_argument("--model-key", required=True, help="Model key in the study JSON.")
    kl.add_argument(
        "--study",
        type=Path,
        help="Study JSON (defaults to multimodel v1 in the source checkout).",
    )
    kl.add_argument("--fit-probe-dir", type=Path, required=True)
    kl.add_argument("--frontier-dir", type=Path, required=True)
    kl.add_argument("--wikitext-path", type=Path, required=True)
    kl.add_argument("--output-dir", type=Path, required=True)
    kl.add_argument("--allow-network", action="store_true")

    gsm = commands.add_parser(
        "evaluate-gsm8k", help="Evaluate the frozen paired GSM8K sample."
    )
    gsm.add_argument("--model-key", required=True, help="Model key in the study JSON.")
    gsm.add_argument(
        "--study",
        type=Path,
        help="Study JSON (defaults to multimodel v1 in the source checkout).",
    )
    gsm.add_argument("--fit-probe-dir", type=Path, required=True)
    gsm.add_argument("--frontier-dir", type=Path, required=True)
    gsm.add_argument("--gsm8k-path", type=Path, required=True)
    gsm.add_argument("--output-dir", type=Path, required=True)
    gsm.add_argument("--allow-network", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-data":
        from .materialize import materialize_study_data

        lock_path = _checkout_path(args.lock, DEFAULT_DATA_LOCK)
        output_dir = _checkout_path(args.output_dir, DEFAULT_DATA_DIR)
        manifest = materialize_study_data(args.source.resolve(), lock_path, output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-study":
        from .config import load_study_config

        study_path = _checkout_path(args.study, DEFAULT_STUDY)
        study = load_study_config(study_path, require_frozen=args.require_frozen)
        print(
            json.dumps(
                {
                    "schema_version": study["schema_version"],
                    "status": study["status"],
                    "scientific_outputs_allowed": study["scientific_outputs_allowed"],
                    "models": list(study["models"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fit-probe":
        from .fit_probe_stage import run_fit_probe_stage

        repository = _repository_root()
        study_path = _checkout_path(args.study, DEFAULT_STUDY, repository=repository)
        data_dir = _checkout_path(
            args.data_dir, DEFAULT_DATA_DIR, repository=repository
        )
        result = run_fit_probe_stage(
            repository=repository,
            study_path=study_path,
            data_dir=data_dir,
            output_dir=args.output_dir.resolve(),
            model_key=args.model_key,
            run_kind=args.run_kind,
            limit=args.limit,
            generation_batch_size=args.generation_batch_size,
            residual_batch_size=args.residual_batch_size,
            local_files_only=not args.allow_network,
        )
        print(
            json.dumps(
                {
                    "model_key": result["model_key"],
                    "run_kind": result["run_kind"],
                    "chosen_estimator": result["layer_selection"]["chosen_estimator"],
                    "chosen_layers": result["layer_selection"]["chosen_layers"],
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "evaluate-frontier":
        from .frontier_stage import run_frontier_stage

        repository = _repository_root()
        study_path = _checkout_path(args.study, DEFAULT_STUDY, repository=repository)
        data_dir = _checkout_path(
            args.data_dir, DEFAULT_DATA_DIR, repository=repository
        )
        result = run_frontier_stage(
            repository=repository,
            study_path=study_path,
            data_dir=data_dir,
            fit_probe_dir=args.fit_probe_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            model_key=args.model_key,
            local_files_only=not args.allow_network,
        )
        print(
            json.dumps(
                {
                    "model_key": result["model_key"],
                    "reporting": result["reporting"],
                    "chosen_estimator": result["chosen_estimator"],
                    "chosen_layers": result["chosen_layers"],
                    "condition_count": result["condition_count"],
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "evaluate-kl":
        from .kl_stage import run_kl_stage

        repository = _repository_root()
        study_path = _checkout_path(args.study, DEFAULT_STUDY, repository=repository)
        result = run_kl_stage(
            repository=repository,
            study_path=study_path,
            fit_probe_dir=args.fit_probe_dir.resolve(),
            frontier_dir=args.frontier_dir.resolve(),
            wikitext_path=args.wikitext_path.resolve(),
            output_dir=args.output_dir.resolve(),
            model_key=args.model_key,
            local_files_only=not args.allow_network,
        )
        print(
            json.dumps(
                {
                    "model_key": result["model_key"],
                    "context_count": result["context_count"],
                    "condition_count": result["condition_count"],
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "evaluate-gsm8k":
        from .gsm8k_stage import run_gsm8k_stage

        repository = _repository_root()
        study_path = _checkout_path(args.study, DEFAULT_STUDY, repository=repository)
        result = run_gsm8k_stage(
            repository=repository,
            study_path=study_path,
            fit_probe_dir=args.fit_probe_dir.resolve(),
            frontier_dir=args.frontier_dir.resolve(),
            gsm8k_path=args.gsm8k_path.resolve(),
            output_dir=args.output_dir.resolve(),
            model_key=args.model_key,
            local_files_only=not args.allow_network,
        )
        print(
            json.dumps(
                {
                    "model_key": result["model_key"],
                    "reporting": result["reporting"],
                    "has_steered_condition": result["condition"] is not None,
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
