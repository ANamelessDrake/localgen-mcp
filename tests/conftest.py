from __future__ import annotations

import os
import shutil
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from localgen_mcp import backend, config  # noqa: E402

FAKE_MAIN = Path(__file__).resolve().parent / "fake_comfy" / "main.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeEnv:
    """A sandboxed LOCALGEN_ROOT whose ComfyUI is tests/fake_comfy/main.py."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = root
        self.comfy_dir = root / "ComfyUI"
        self.comfy_dir.mkdir(parents=True)
        shutil.copy(FAKE_MAIN, self.comfy_dir / "main.py")
        self.port = free_port()
        env = {
            "LOCALGEN_ROOT": str(root),
            "LOCALGEN_COMFY_PYTHON": sys.executable,
            "LOCALGEN_PORT": str(self.port),
            "LOCALGEN_POLL_INTERVAL": "1",
            "LOCALGEN_STOP_GRACE": "5",
            "LOCALGEN_START_TIMEOUT": "30",
            "LOCALGEN_IDLE_TIMEOUT_MIN": "1",
            "LOCALGEN_IDLE_TIMEOUT_MAX": "600",
            "LOCALGEN_IDLE_TIMEOUT": "300",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        self.settings = config.load()

    def launches(self) -> list[str]:
        f = self.comfy_dir / "launches.log"
        return f.read_text().splitlines() if f.exists() else []

    def set_busy(self, busy: bool) -> None:
        f = self.comfy_dir / "busy"
        if busy:
            f.touch()
        else:
            f.unlink(missing_ok=True)

    def wait_until(self, predicate, timeout: float, interval: float = 0.2) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return predicate()

    def running(self) -> bool:
        return backend.ComfyClient(self.settings.base_url).alive(timeout=1)

    def teardown(self) -> None:
        pid = backend.supervisor_pid(self.settings)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.wait_until(lambda: backend.supervisor_pid(self.settings) is None and not self.running(), 15)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    env = FakeEnv(tmp_path / "root", monkeypatch)
    yield env
    env.teardown()
