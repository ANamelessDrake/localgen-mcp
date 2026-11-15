"""Workflow registry.

Each workflow module defines one WORKFLOW (a Workflow instance): the modality
it serves, the model files it needs (relative to the ComfyUI models folder),
its default parameters and a build() function that returns a ComfyUI API
format graph. Tools look workflows up here by modality, so no model name is
hardcoded outside this package.

To add a model: add a module with a WORKFLOW, list it in _MODULES, add its
files to setup/models.manifest. To add a modality: also register a tool in
server.py that calls run_workflow() with the new modality.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_MODULES = ["zimage_turbo", "flux2_klein_4b", "wan22_ti2v_5b", "trellis2", "pixal3d_multiview", "texture_mesh", "part_isolate"]


@dataclass(frozen=True)
class Workflow:
    name: str
    modality: str                      # image | video | mesh | texture
    description: str
    models: list[str]                  # files under the ComfyUI models dir
    defaults: dict[str, Any]
    build: Callable[[dict[str, Any]], dict]   # params (defaults merged) -> API graph
    # Named input image slots this workflow accepts (for example ("start_image",)
    # or ("front", "left", "back", "right")), and how many must be filled.
    image_slots: tuple[str, ...] = ()
    min_images: int = 0
    is_default: bool = False
    limits: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Takes a 3D model file (uploaded to ComfyUI's input/3d) as params["mesh"].
    needs_mesh: bool = False

    @property
    def image_input(self) -> str:
        if not self.image_slots:
            return "none"
        return "required" if self.min_images else "optional"

    def accepts(self, slots: set[str]) -> bool:
        return slots <= set(self.image_slots) and len(slots) >= self.min_images

    def missing_models(self, models_dir: Path) -> list[str]:
        return [m for m in self.models if not (models_dir / m).exists()]

    def params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Defaults with non-None overrides applied and numeric limits enforced."""
        p = dict(self.defaults)
        p.update({k: v for k, v in overrides.items() if v is not None})
        for key, (lo, hi) in self.limits.items():
            if key in p and p[key] is not None:
                v = int(p[key])
                if not lo <= v <= hi:
                    raise ValueError(f"{key}={v} is outside the allowed range {lo}..{hi} for {self.name}")
                p[key] = v
        return p


def _load() -> dict[str, Workflow]:
    found = {}
    for mod in _MODULES:
        wf = importlib.import_module(f"{__name__}.{mod}").WORKFLOW
        found[wf.name] = wf
    return found


REGISTRY: dict[str, Workflow] = _load()


def for_modality(modality: str) -> list[Workflow]:
    return [w for w in REGISTRY.values() if w.modality == modality]


def pick(modality: str, name: str | None = None, slots: set[str] | frozenset = frozenset()) -> Workflow:
    """The workflow to run: the named one, else the modality default if it accepts
    these image slots, else the first registered workflow that does."""
    slots = set(slots)
    if name:
        wf = get(modality, name)
        if not wf.accepts(slots):
            raise ValueError(f"{wf.name} takes image slots {list(wf.image_slots)} (at least {wf.min_images}); "
                             f"got {sorted(slots)}")
        return wf
    options = sorted(for_modality(modality), key=lambda w: not w.is_default)
    for w in options:
        if w.accepts(slots):
            return w
    raise ValueError(f"no {modality} workflow accepts image slots {sorted(slots)}; "
                     f"available: {[(w.name, list(w.image_slots)) for w in options]}")


def get(modality: str, name: str | None = None) -> Workflow:
    options = for_modality(modality)
    if name:
        for w in options:
            if w.name == name:
                return w
        raise ValueError(f"no {modality} workflow named {name!r}; available: {[w.name for w in options]}")
    for w in options:
        if w.is_default:
            return w
    if options:
        return options[0]
    raise ValueError(f"no workflow registered for modality {modality!r}")
