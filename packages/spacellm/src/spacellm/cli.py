"""``spacellm`` CLI entry point.

Built on Typer for argument parsing and Rich for output. v0.1 ships
``version`` / ``runs`` / ``show`` / ``profile``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from rich.table import Table

from spacellm import __version__
from spacellm.observability import RunDB
from spacellm.profiling import state_dict_profile

app = typer.Typer(
    name="spacellm",
    help="SpaceLLM, radiation-tolerant transformer framework.",
    add_completion=False,
    no_args_is_help=True,
)

_console = Console()


@app.command(name="version")
def version_cmd() -> None:
    """Print the installed framework version."""
    typer.echo(__version__)


@app.command(name="runs")
def list_runs(
    db: Annotated[
        Path,
        typer.Option("--db", "-d", help="Path to the runs database."),
    ] = Path("runs.db"),
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", min=1, help="Maximum rows to list."),
    ] = 20,
) -> None:
    """List the most recent runs."""
    if not db.exists():
        _console.print(f"[yellow]No run database at {db}[/yellow]")
        raise typer.Exit(code=0)
    with RunDB(db) as store:
        rows = store.list_runs(limit=limit)
    if not rows:
        _console.print("[dim]no runs[/dim]")
        return
    table = Table(title=f"Runs in {db}")
    table.add_column("run_id", style="cyan", no_wrap=True)
    table.add_column("name")
    table.add_column("model")
    table.add_column("started")
    table.add_column("ended")
    table.add_column("flips", justify="right", style="magenta")
    for r in rows:
        table.add_row(
            r.run_id[:8],
            r.name or "-",
            r.model_name or "-",
            r.started_at.isoformat(timespec="seconds"),
            r.ended_at.isoformat(timespec="seconds") if r.ended_at else "-",
            str(r.n_faults_injected),
        )
    _console.print(table)


@app.command(name="show")
def show_run(
    run_id: Annotated[str, typer.Argument(help="Run id (or its 8-char prefix).")],
    db: Annotated[
        Path,
        typer.Option("--db", "-d", help="Path to the runs database."),
    ] = Path("runs.db"),
) -> None:
    """Show details for one run."""
    if not db.exists():
        _console.print(f"[red]No run database at {db}[/red]")
        raise typer.Exit(code=1)
    with RunDB(db) as store:
        full_id = _resolve_run_id(store, run_id)
        if full_id is None:
            _console.print(f"[red]No run matches '{run_id}'[/red]")
            raise typer.Exit(code=1)
        record = store.get_run(full_id)
        events = store.fault_events_for(full_id)
    assert record is not None
    _console.print(f"[bold]run_id[/bold] {record.run_id}")
    _console.print(f"[bold]name[/bold]   {record.name or '-'}")
    _console.print(f"[bold]model[/bold]  {record.model_name or '-'}")
    _console.print(f"[bold]started[/bold] {record.started_at.isoformat(timespec='seconds')}")
    _console.print(
        f"[bold]ended[/bold]  "
        f"{record.ended_at.isoformat(timespec='seconds') if record.ended_at else '-'}",
    )
    _console.print(
        f"[bold]flips[/bold]   {record.n_faults_injected} ({record.n_faults_recovered} recovered)",
    )
    if record.metrics:
        _console.print("[bold]metrics[/bold]:")
        for k, v in record.metrics.items():
            _console.print(f"  {k} = {v}")
    if events:
        _console.print(f"[bold]fault events[/bold]: {len(events)}")


def _resolve_run_id(store: RunDB, candidate: str) -> str | None:
    """Resolve a full run id from a (possibly 8-char) prefix."""
    for record in store.list_runs(limit=1000):
        if record.run_id == candidate or record.run_id.startswith(candidate):
            return record.run_id
    return None


# ────────────────────────────────────────────────────────────────────
# `spacellm profile`
# ────────────────────────────────────────────────────────────────────


def _load_state_dict_from_path(path: Path) -> dict[str, torch.Tensor]:
    """Load a state-dict from safetensors / .pt / .pth / .bin / HF dir."""
    if path.is_dir():
        return _load_hf_dir(path)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return _load_safetensors(path)
    if suffix in {".pt", ".pth", ".bin", ".ckpt"}:
        return _load_torch(path)
    raise typer.BadParameter(
        f"Unsupported file type '{suffix}'. Expected .safetensors / .pt / "
        ".pth / .bin / .ckpt or a HF model directory.",
    )


def _load_hf_dir(path: Path) -> dict[str, torch.Tensor]:
    shards = sorted(path.glob("*.safetensors"))
    if shards:
        merged: dict[str, torch.Tensor] = {}
        for shard in shards:
            merged.update(_load_safetensors(shard))
        return merged
    legacy = path / "pytorch_model.bin"
    if legacy.exists():
        return _load_torch(legacy)
    raise typer.BadParameter(
        f"No '*.safetensors' or 'pytorch_model.bin' found under {path}",
    )


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter(
            "safetensors is not installed. Run: pip install 'spacellm[hf]'",
        ) from exc
    return dict(load_file(str(path)))


def _load_torch(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        return {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
    raise typer.BadParameter(
        f"File at {path} did not load as a state-dict (got {type(loaded).__name__}).",
    )


@app.command(name="profile")
def profile_cmd(
    model_path: Annotated[
        Path,
        typer.Argument(
            help="HF model directory, safetensors file, or torch state-dict.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to write the report."),
    ] = Path("profile.html"),
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: 'html' or 'json'."),
    ] = "html",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", min=1, help="Top-K layers to print."),
    ] = 20,
) -> None:
    """Profile a saved transformer, rank layers by parameter count, write a report."""
    if fmt not in {"html", "json"}:
        raise typer.BadParameter("--format must be 'html' or 'json'")
    if not model_path.exists():
        raise typer.BadParameter(f"No file or directory at {model_path}")

    state_dict = _load_state_dict_from_path(model_path)
    if not state_dict:
        raise typer.BadParameter(f"No tensors found in {model_path}")

    profile = state_dict_profile(state_dict)
    title = f"Profile · {model_path.name}"

    if fmt == "html":
        out_path = profile.save_html(output, title=title)
    else:
        out_path = profile.save_json(output)

    table = Table(title=f"Top {top_k} layers · {model_path.name}")
    table.add_column("#", justify="right")
    table.add_column("layer")
    table.add_column("dtype")
    table.add_column("params", justify="right", style="magenta")
    table.add_column("bits", justify="right")
    for i, layer in enumerate(profile.top_k(top_k), 1):
        table.add_row(
            str(i),
            layer.layer_name,
            layer.module_class,
            f"{layer.n_parameters:,}",
            f"{layer.n_bits:,}",
        )
    _console.print(table)
    _console.print(
        f"\n[bold]layers:[/bold] {len(profile.layers):,}    "
        f"[bold]total params:[/bold] {sum(layer.n_parameters for layer in profile.layers):,}",
    )
    _console.print(f"[green]wrote {out_path}[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
