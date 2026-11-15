"""Settings, all overridable by environment variable.

Everything large (ComfyUI, weights, venvs, caches, logs, outputs, run state)
lives under LOCALGEN_ROOT, which defaults to /ai so nothing lands on /home.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Settings:
    root: Path
    comfy_dir: Path
    comfy_python: Path
    host: str
    port: int
    run_dir: Path
    log_dir: Path
    output_dir: Path
    idle_timeout: int
    idle_timeout_min: int
    idle_timeout_max: int
    start_timeout: int
    poll_interval: int
    stop_grace: int
    comfy_extra_args: list[str] = field(default_factory=list)
    blender: str | None = None  # None: find "blender" on PATH
    blender_threads: int = 8
    retopo_timeout: int = 1800

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # Run state files, shared by every MCP server process and the supervisor.
    @property
    def start_lock(self) -> Path:
        return self.run_dir / "start.lock"

    @property
    def supervisor_lock(self) -> Path:
        return self.run_dir / "supervisor.lock"

    @property
    def pidfile(self) -> Path:
        return self.run_dir / "supervisor.pid"

    @property
    def activity_file(self) -> Path:
        return self.run_dir / "last_activity"

    @property
    def timeout_file(self) -> Path:
        return self.run_dir / "idle_timeout"

    @property
    def supervisor_log(self) -> Path:
        return self.log_dir / "localgen-supervisor.log"

    @property
    def comfy_log(self) -> Path:
        return self.log_dir / "localgen-comfyui.log"

    def clamp_timeout(self, seconds: int | None) -> int:
        """Requested idle timeout bounded to [min, max]; None means the default."""
        value = self.idle_timeout if seconds is None else int(seconds)
        return max(self.idle_timeout_min, min(self.idle_timeout_max, value))


def load() -> Settings:
    root = _env_path("LOCALGEN_ROOT", Path("/ai"))
    comfy_dir = _env_path("LOCALGEN_COMFY_DIR", root / "ComfyUI")
    host = os.environ.get("LOCALGEN_HOST", "127.0.0.1")
    # Hard requirement: the managed backend is reachable from this machine only.
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError(f"LOCALGEN_HOST must be a loopback address, got {host!r}")
    tmin = _env_int("LOCALGEN_IDLE_TIMEOUT_MIN", 60)
    tmax = _env_int("LOCALGEN_IDLE_TIMEOUT_MAX", 7200)
    if tmin < 1 or tmax < tmin:
        raise ValueError("idle timeout bounds must satisfy 1 <= min <= max")
    default_timeout = max(tmin, min(tmax, _env_int("LOCALGEN_IDLE_TIMEOUT", 300)))
    return Settings(
        root=root,
        comfy_dir=comfy_dir,
        comfy_python=_env_path("LOCALGEN_COMFY_PYTHON", comfy_dir / ".venv" / "bin" / "python"),
        host=host,
        port=_env_int("LOCALGEN_PORT", 8188),
        run_dir=_env_path("LOCALGEN_RUN_DIR", root / "run" / "localgen"),
        log_dir=_env_path("LOCALGEN_LOG_DIR", root / "logs"),
        output_dir=_env_path("LOCALGEN_OUTPUT_DIR", root / "output" / "localgen"),
        idle_timeout=default_timeout,
        idle_timeout_min=tmin,
        idle_timeout_max=tmax,
        start_timeout=_env_int("LOCALGEN_START_TIMEOUT", 180),
        poll_interval=_env_int("LOCALGEN_POLL_INTERVAL", 5),
        stop_grace=_env_int("LOCALGEN_STOP_GRACE", 20),
        comfy_extra_args=os.environ.get("LOCALGEN_COMFY_ARGS", "").split(),
        blender=os.environ.get("LOCALGEN_BLENDER") or None,
        blender_threads=max(1, _env_int("LOCALGEN_BLENDER_THREADS", 8)),
        retopo_timeout=_env_int("LOCALGEN_TIMEOUT_RETOPO", 1800),
    )
