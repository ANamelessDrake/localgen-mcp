"""Client side of the shared backend: start on demand, share, extend, stop.

Every MCP server process (one per agent session) calls ensure_backend() before
submitting work. A file lock serialises startup so concurrent callers start
exactly one supervisor, and that supervisor owns the single ComfyUI process.

Run state lives in Settings.run_dir:
    start.lock       held while starting, and by the supervisor while shutting down
    supervisor.lock  held by the live supervisor for its whole life
    supervisor.pid   pid of the live supervisor
    last_activity    touched by clients; its mtime resets the idle clock
    idle_timeout     effective idle timeout in seconds, reread by the supervisor
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .comfy import ComfyClient, ComfyError
from .config import Settings

logger = logging.getLogger("localgen_mcp.backend")


class BackendError(RuntimeError):
    pass


@dataclass
class BackendInfo:
    url: str
    managed: bool              # False when someone else's ComfyUI (for example run.sh) is on the port
    started: bool              # True when this call started the backend
    idle_timeout_seconds: int | None  # effective timeout; None when not managed
    supervisor_pid: int | None

    def as_dict(self) -> dict:
        return {
            "backend_url": self.url,
            "backend_managed": self.managed,
            "backend_started_by_this_call": self.started,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "supervisor_pid": self.supervisor_pid,
        }


# -- small file helpers ---------------------------------------------------


@contextlib.contextmanager
def flocked(path: Path, timeout: float | None = None) -> Iterator[None]:
    """Exclusive flock on path; blocks (optionally up to timeout seconds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if timeout is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise BackendError(f"timed out after {timeout}s waiting for lock {path}")
                    time.sleep(0.2)
        yield
    finally:
        os.close(fd)  # closing releases the lock


def touch_activity(settings: Settings) -> None:
    settings.activity_file.parent.mkdir(parents=True, exist_ok=True)
    settings.activity_file.touch()
    os.utime(settings.activity_file, None)


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def write_int_atomic(path: Path, value: int) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(f"{value}\n")
    os.replace(tmp, path)


def supervisor_pid(settings: Settings) -> int | None:
    """Pid of the live supervisor, or None. Liveness is the supervisor lock, not the pidfile."""
    pid = read_int(settings.pidfile)
    if pid is None:
        return None
    fd = os.open(settings.supervisor_lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return pid  # somebody holds it: the supervisor is alive
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def request_timeout(settings: Settings, seconds: int | None) -> int:
    """Record an idle timeout request against the running supervisor.

    The effective value only ever grows while a backend is up (maximum of the
    requests made since it started), so one agent cannot cut short another
    agent's longer request. Returns the effective value.
    """
    with flocked(settings.run_dir / "idle_timeout.lock"):
        current = read_int(settings.timeout_file) or settings.idle_timeout
        wanted = settings.clamp_timeout(seconds) if seconds is not None else current
        effective = max(current, wanted)
        if effective != current or not settings.timeout_file.exists():
            write_int_atomic(settings.timeout_file, effective)
        return effective


# -- start / ensure -------------------------------------------------------


def _spawn_supervisor(settings: Settings, idle_timeout: int) -> subprocess.Popen:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log = open(settings.supervisor_log, "ab")
    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "localgen_mcp.supervisor", "--idle-timeout", str(idle_timeout)]
    logger.info("starting supervisor: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                            start_new_session=True, close_fds=True, cwd=str(settings.run_dir))


def _start_fresh(settings: Settings, idle_timeout_seconds: int | None) -> subprocess.Popen:
    requested = settings.clamp_timeout(idle_timeout_seconds)
    # Fresh backend: the starter's request is the effective value.
    write_int_atomic(settings.timeout_file, requested)
    touch_activity(settings)
    return _spawn_supervisor(settings, requested)


def ensure_backend(settings: Settings, idle_timeout_seconds: int | None = None) -> BackendInfo:
    """Make sure a ComfyUI backend answers on the configured port; start one if needed."""
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(settings.base_url)
    # The supervisor holds start.lock while it shuts down, so waiting here can
    # never race a dying server. Allow for a slow shutdown plus a full start.
    with flocked(settings.start_lock, timeout=settings.start_timeout + settings.stop_grace + 60):
        if client.alive():
            pid = supervisor_pid(settings)
            if pid is None:
                # Not ours (for example one started by hand). Use it, do not manage it.
                return BackendInfo(settings.base_url, False, False, None, None)
            effective = request_timeout(settings, idle_timeout_seconds)
            touch_activity(settings)
            return BackendInfo(settings.base_url, True, False, effective, pid)

        proc = None
        if supervisor_pid(settings) is None:
            proc = _start_fresh(settings, idle_timeout_seconds)
        # else: a supervisor exists but its ComfyUI does not answer yet (its
        # starter gave up or crashed). Wait for it rather than start a second one.

        deadline = time.monotonic() + settings.start_timeout
        while time.monotonic() < deadline:
            if client.alive():
                touch_activity(settings)
                effective = request_timeout(settings, idle_timeout_seconds) if proc is None \
                    else read_int(settings.timeout_file)
                return BackendInfo(settings.base_url, True, proc is not None, effective,
                                   supervisor_pid(settings))
            if proc is not None and proc.poll() is not None:
                raise BackendError(f"supervisor exited with code {proc.returncode} before ComfyUI came up; "
                                   f"see {settings.supervisor_log} and {settings.comfy_log}\n"
                                   + _tail(settings.comfy_log))
            if proc is None and supervisor_pid(settings) is None:
                proc = _start_fresh(settings, idle_timeout_seconds)
            time.sleep(1)
        raise BackendError(f"ComfyUI did not answer within {settings.start_timeout}s; see {settings.comfy_log}\n"
                           + _tail(settings.comfy_log))


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


# -- status / stop --------------------------------------------------------


def status(settings: Settings) -> dict:
    client = ComfyClient(settings.base_url)
    out: dict = {"backend_url": settings.base_url, "running": False}
    pid = supervisor_pid(settings)
    out["backend_managed"] = pid is not None
    out["supervisor_pid"] = pid
    out["default_idle_timeout_seconds"] = settings.idle_timeout
    out["idle_timeout_bounds_seconds"] = [settings.idle_timeout_min, settings.idle_timeout_max]
    out["idle_timeout_seconds"] = read_int(settings.timeout_file) if pid else None
    try:
        out["last_activity_seconds_ago"] = round(time.time() - settings.activity_file.stat().st_mtime, 1)
    except FileNotFoundError:
        out["last_activity_seconds_ago"] = None
    try:
        stats = client.system_stats()
        out["running"] = True
        out["queue_length"] = client.queue_length()
        out["devices"] = [
            {"name": d.get("name"), "vram_total_mb": round(d.get("vram_total", 0) / 2**20),
             "vram_free_mb": round(d.get("vram_free", 0) / 2**20)}
            for d in stats.get("devices", [])
        ]
        out["comfyui_version"] = stats.get("system", {}).get("comfyui_version")
    except ComfyError:
        out["queue_length"] = 0
    out["gpu"] = nvidia_smi()
    return out


def nvidia_smi() -> dict | None:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        name, used, total, util = [x.strip() for x in raw.split(",")]
        return {"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization_pct": int(util)}
    except Exception:  # nvidia-smi missing or failing should not break status
        return None


def stop_backend(settings: Settings, force: bool = False, wait: float = 60) -> dict:
    """Ask the supervisor to shut ComfyUI down now.

    Refuses while jobs are queued or running unless force=True, so one agent
    does not kill another agent's work.
    """
    pid = supervisor_pid(settings)
    client = ComfyClient(settings.base_url)
    if pid is None:
        running = client.alive()
        return {"stopped": False, "running": running,
                "reason": "an unmanaged ComfyUI is running on the port; it was not started by localgen and was left alone"
                if running else "backend was not running"}
    if not force:
        try:
            qlen = client.queue_length()
        except ComfyError:
            qlen = 0
        if qlen:
            return {"stopped": False, "running": True, "queue_length": qlen,
                    "reason": "jobs are queued or running; pass force=true to stop anyway"}
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if supervisor_pid(settings) is None and not client.alive():
            return {"stopped": True, "running": False}
        time.sleep(0.5)
    return {"stopped": False, "running": client.alive(), "reason": f"supervisor {pid} still alive after {wait}s"}
