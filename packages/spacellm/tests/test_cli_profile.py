"""Tests for the ``spacellm profile`` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from spacellm.cli import app

runner = CliRunner()


def _save_state_dict(tmp_path: Path, name: str = "tiny.pt") -> Path:
    sd = {
        "embed.weight": torch.zeros(50, 8),
        "head.weight": torch.zeros(8, 50),
        "head.bias": torch.zeros(50),
    }
    out = tmp_path / name
    torch.save(sd, out)
    return out


def test_profile_pt_file_writes_html(tmp_path: Path) -> None:
    src = _save_state_dict(tmp_path, "model.pt")
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["profile", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    text = out.read_text()
    assert "<!DOCTYPE html>" in text
    assert "embed.weight" in text


def test_profile_writes_json(tmp_path: Path) -> None:
    src = _save_state_dict(tmp_path, "model.pt")
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["profile", str(src), "-f", "json", "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert "embed.weight" in out.read_text()


def test_profile_safetensors_path(tmp_path: Path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    sd = {"a.weight": torch.zeros(8, 16), "b.bias": torch.zeros(16)}
    src = tmp_path / "model.safetensors"
    safetensors_torch.save_file(sd, str(src))
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["profile", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "a.weight" in out.read_text()


def test_profile_hf_directory_with_safetensors(tmp_path: Path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sd = {"layers.0.weight": torch.zeros(4, 4), "layers.0.bias": torch.zeros(4)}
    safetensors_torch.save_file(sd, str(model_dir / "model.safetensors"))
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["profile", str(model_dir), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "layers.0.weight" in out.read_text()


def test_profile_unknown_path(tmp_path: Path) -> None:
    result = runner.invoke(app, ["profile", str(tmp_path / "nope.pt")])
    assert result.exit_code != 0


def test_profile_unknown_format(tmp_path: Path) -> None:
    src = _save_state_dict(tmp_path)
    result = runner.invoke(app, ["profile", str(src), "-f", "csv"])
    assert result.exit_code != 0


def test_profile_invalid_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "weights.foo"
    bogus.write_bytes(b"not a real file")
    result = runner.invoke(app, ["profile", str(bogus)])
    assert result.exit_code != 0
