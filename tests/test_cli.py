# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from sycophancy_steering import __version__
from sycophancy_steering.cli import main

REPOSITORY = Path(__file__).resolve().parents[1]


def test_validate_draft_command(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "validate-study",
            "--study",
            str(REPOSITORY / "configs" / "studies" / "multimodel_v1.json"),
        ]
    )

    assert result == 0
    assert '"status": "draft"' in capsys.readouterr().out


def test_validate_frozen_rejects_current_draft() -> None:
    with pytest.raises(ValueError, match="not frozen"):
        main(
            [
                "validate-study",
                "--study",
                str(REPOSITORY / "configs" / "studies" / "multimodel_v1.json"),
                "--require-frozen",
            ]
        )


def test_help_works_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "fit-probe" in output
    assert "evaluate-frontier" in output


def test_version_works_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(["--version"])

    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"sycophancy-steering {__version__}"


def test_explicit_study_validation_works_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    result = main(
        [
            "validate-study",
            "--study",
            str(REPOSITORY / "configs" / "studies" / "multimodel_v1.json"),
        ]
    )

    assert result == 0
    assert '"status": "draft"' in capsys.readouterr().out
