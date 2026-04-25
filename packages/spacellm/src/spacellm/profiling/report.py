"""Profile result types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LayerSensitivity:
    """Sensitivity record for one module.

    Attributes:
        layer_name: Dotted path emitted by ``named_modules``.
        module_class: Concrete Python class name, e.g. ``"Linear"``.
        n_parameters: Sum of ``numel()`` over directly-owned parameters
            (``recurse=False``).
        n_bits: ``n_parameters`` × element-size × 8.
        mean_perturbation: Mean MSE between clean and post-flip outputs
            for this module. Populated only by :func:`sensitivity_profile`.
        max_perturbation: Maximum MSE seen across injected flips.
            Populated only by :func:`sensitivity_profile`.
    """

    layer_name: str
    module_class: str
    n_parameters: int
    n_bits: int
    mean_perturbation: float | None = None
    max_perturbation: float | None = None


@dataclass(slots=True)
class ProfileReport:
    """A profile snapshot.

    The ``layers`` list is *unsorted*, call :meth:`top_k` to retrieve a
    ranking by a chosen criterion.
    """

    layers: list[LayerSensitivity] = field(default_factory=list)

    def top_k(
        self,
        k: int,
        *,
        by: str = "n_parameters",
    ) -> list[LayerSensitivity]:
        """Return the ``k`` heaviest layers by the requested attribute.

        Args:
            k: Maximum number of layers to return.
            by: One of ``"n_parameters"``, ``"n_bits"``,
                ``"mean_perturbation"``, ``"max_perturbation"``. Layers
                with ``None`` for the chosen attribute are filtered out.

        Returns:
            Sorted descending. Length ≤ ``k``.
        """
        valid = [layer for layer in self.layers if getattr(layer, by) is not None]
        valid.sort(key=lambda layer: -getattr(layer, by))
        return valid[:k]

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        """JSON-serialisable dict representation."""
        return {"layers": [asdict(layer) for layer in self.layers]}

    def save_json(self, path: str | Path) -> Path:
        """Write the profile to a JSON file. Returns the resolved path."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def to_html(self, *, title: str = "SpaceLLM Profile") -> str:
        """Render the profile as a self-contained HTML document."""
        from spacellm.observability.report import render_profile_html  # noqa: PLC0415

        return render_profile_html(self, title=title)

    def save_html(self, path: str | Path, *, title: str = "SpaceLLM Profile") -> Path:
        """Write the profile as HTML. Returns the resolved path."""
        path = Path(path)
        path.write_text(self.to_html(title=title))
        return path


__all__ = ["LayerSensitivity", "ProfileReport"]
