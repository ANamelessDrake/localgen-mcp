"""Detached supervisor that owns the shared ComfyUI process.

Started by backend.ensure_backend() in its own session. It launches ComfyUI
bound to localhost with --offline, then polls. When the queue is empty and no
client has touched the activity file for the effective idle timeout (reread
from the control file on every poll), it takes start.lock so no client can
submit into a dying server, terminates ComfyUI (SIGTERM, then SIGKILL after a
grace period) and exits. Killing the process is what frees the CUDA context
and all of its RAM.

    python -m localgen_mcp.supervisor --idle-timeout 300
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import logging
import os
import signal
import subprocess
import sys
import time

from . import config
from .backend import flocked, read_int, write_int_atomic
from .comfy import ComfyClient, ComfyError

logger = logging.getLogger("localgen_mcp.supervisor")

PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Run in the ComfyUI child before exec: get SIGTERM if the supervisor dies."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except OSError:
        pass


def comfy_command(s: config.Settings) -> list[str]:
    return [str(s.comfy_python), "main.py",
            "--listen", s.host, "--port", str(s.port),
            "--offline",
            "--disable-auto-launch",
            *s.comfy_extra_args]


def comfy_env(s: config.Settings) -> dict:
    env = dict(os.environ)
    # Same redirection as /ai/ComfyUI/run.sh: keep caches and temp files off /home and /tmp.
    env.update({
        "UV_CACHE_DIR": str(s.root / ".cache" / "uv"),
        "PIP_CACHE_DIR": str(s.root / ".cache" / "pip"),
        "HF_HOME": str(s.root / ".cache" / "huggingface"),
        "TORCH_HOME": str(s.root / ".cache" / "torch"),
        "XDG_CACHE_HOME": str(s.root / ".cache"),
        "TMPDIR": str(s.root / "tmp"),
        # Belt and braces for offline operation: no hub lookups from any library.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    })
    env.pop("PYTHONPATH", None)
    return env


class Supervisor:
    def __init__(self, s: config.Settings, idle_timeout: int, lock_fd: int):
        self.s = s
        self.lock_fd = lock_fd
        self.idle_timeout = s.clamp_timeout(idle_timeout)
        self.client = ComfyClient(s.base_url, timeout=10)
        self.proc: subprocess.Popen | None = None
        self.stop_requested = False
        self.last_busy = time.time()

    def _on_signal(self, signum, _frame):
        logger.info("received %s, shutting down", signal.Signals(signum).name)
        self.stop_requested = True

    def effective_timeout(self) -> int:
        value = read_int(self.s.timeout_file)
        return self.s.clamp_timeout(value) if value is not None else self.idle_timeout

    def idle_for(self) -> float:
        try:
            last_activity = self.s.activity_file.stat().st_mtime
        except FileNotFoundError:
            last_activity = 0.0
        return time.time() - max(last_activity, self.last_busy)

    def queue_busy(self) -> bool:
        try:
            return self.client.queue_length() > 0
        except ComfyError:
            return False

    def cleanup(self) -> None:
        """Remove the pidfile and timeout file and release the singleton lock (idempotent)."""
        if self.lock_fd < 0:
            return
        for f in (self.s.pidfile, self.s.timeout_file):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        os.close(self.lock_fd)
        self.lock_fd = -1
        logger.info("run state cleared")

    def start_comfy(self) -> None:
        for d in (self.s.log_dir, self.s.root / "tmp", self.s.root / ".cache"):
            d.mkdir(parents=True, exist_ok=True)
        log = open(self.s.comfy_log, "ab")
        cmd = comfy_command(self.s)
        logger.info("launching ComfyUI: %s (cwd %s)", " ".join(cmd), self.s.comfy_dir)
        self.proc = subprocess.Popen(cmd, cwd=str(self.s.comfy_dir), env=comfy_env(self.s),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     preexec_fn=_die_with_parent)
        logger.info("ComfyUI pid %d", self.proc.pid)

    def stop_comfy(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        logger.info("terminating ComfyUI pid %d", self.proc.pid)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=self.s.stop_grace)
        except subprocess.TimeoutExpired:
            logger.warning("ComfyUI ignored SIGTERM for %ss, killing", self.s.stop_grace)
            self.proc.kill()
            self.proc.wait()
        logger.info("ComfyUI exited with %s", self.proc.returncode)

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGHUP, self._on_signal)
        self.start_comfy()
        try:
            return self._loop()
        finally:
            self.stop_comfy()

    def _loop(self) -> int:
        reason = "unknown"
        while True:
            if self.proc.poll() is not None:
                logger.error("ComfyUI exited on its own with code %s", self.proc.returncode)
                return 1
            if self.stop_requested:
                reason = "stop requested"
            else:
                if self.queue_busy():
                    self.last_busy = time.time()
                timeout = self.effective_timeout()
                if self.idle_for() < timeout:
                    time.sleep(self.s.poll_interval)
                    continue
                reason = f"idle for {self.idle_for():.0f}s (timeout {timeout}s)"
            # Block new clients, then re-check: a client may have touched the
            # activity file just before we took the lock.
            with flocked(self.s.start_lock):
                if not self.stop_requested:
                    if self.queue_busy():
                        self.last_busy = time.time()
                    if self.idle_for() < self.effective_timeout() or self.queue_busy():
                        logger.info("activity arrived during shutdown check; staying up")
                        continue
                logger.info("shutting down: %s", reason)
                self.stop_comfy()
                # Release run state while still holding start.lock, so the next
                # client sees a clean "nothing running" state, never a half-dead one.
                self.cleanup()
                return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="localgen ComfyUI supervisor")
    ap.add_argument("--idle-timeout", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [%(levelname)s] supervisor[%(process)d]: %(message)s")
    s = config.load()
    s.run_dir.mkdir(parents=True, exist_ok=True)

    # Singleton: hold the supervisor lock for our whole life.
    lock_fd = os.open(s.supervisor_lock, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + 5
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                logger.error("another supervisor holds %s; exiting", s.supervisor_lock)
                return 2
            time.sleep(0.2)

    write_int_atomic(s.pidfile, os.getpid())
    sup = Supervisor(s, a.idle_timeout if a.idle_timeout is not None else s.idle_timeout, lock_fd)
    if read_int(s.timeout_file) is None:
        write_int_atomic(s.timeout_file, sup.idle_timeout)
    logger.info("supervisor up, idle timeout %ss", sup.effective_timeout())
    try:
        return sup.run()
    finally:
        sup.cleanup()
        logger.info("supervisor exiting")


if __name__ == "__main__":
    sys.exit(main())
