"""Supervisor and backend lifecycle against a fake ComfyUI (no GPU, a few seconds each)."""

from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import time

from localgen_mcp import backend, config


def test_cold_start_launches_localhost_offline(fake):
    info = backend.ensure_backend(fake.settings, 120)
    assert info.managed and info.started
    assert info.idle_timeout_seconds == 120
    assert fake.running()
    (argv,) = [json.loads(line) for line in fake.launches()]
    # Hard requirements: loopback only, always offline.
    assert argv[argv.index("--listen") + 1] == "127.0.0.1"
    assert "--offline" in argv
    assert "0.0.0.0" not in argv


def test_second_call_reuses_backend(fake):
    backend.ensure_backend(fake.settings)
    info = backend.ensure_backend(fake.settings)
    assert info.managed and not info.started
    assert len(fake.launches()) == 1


def test_timeout_only_grows_while_running(fake):
    assert backend.ensure_backend(fake.settings, 100).idle_timeout_seconds == 100
    assert backend.ensure_backend(fake.settings, 250).idle_timeout_seconds == 250
    # A shorter request from another agent never shortens it.
    assert backend.ensure_backend(fake.settings, 50).idle_timeout_seconds == 250
    # No request keeps the current value.
    assert backend.ensure_backend(fake.settings, None).idle_timeout_seconds == 250
    assert backend.status(fake.settings)["idle_timeout_seconds"] == 250


def test_timeout_is_clamped_to_bounds(fake):
    assert backend.ensure_backend(fake.settings, 10**6).idle_timeout_seconds == fake.settings.idle_timeout_max


def test_idle_shutdown_on_custom_timeout(fake):
    t0 = time.monotonic()
    backend.ensure_backend(fake.settings, 3)
    assert fake.wait_until(lambda: not fake.running(), 15)
    elapsed = time.monotonic() - t0
    assert 3 <= elapsed < 10, elapsed
    assert backend.supervisor_pid(fake.settings) is None
    assert not fake.settings.pidfile.exists()
    assert not fake.settings.timeout_file.exists()


def test_extension_from_second_client_is_honoured(fake):
    t0 = time.monotonic()
    backend.ensure_backend(fake.settings, 3)
    # A second client (separate process, as another MCP server would be) extends to 8 s.
    code = ("import sys; sys.path.insert(0, %r); from localgen_mcp import backend, config; "
            "print(backend.ensure_backend(config.load(), 8).idle_timeout_seconds)") % str(backend.__file__.rsplit("/", 2)[0])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "8"
    time.sleep(max(0, 5 - (time.monotonic() - t0)))
    assert fake.running(), "backend stopped on the original 3 s timeout despite the extension"
    assert fake.wait_until(lambda: not fake.running(), 15)
    assert time.monotonic() - t0 >= 8


def test_busy_queue_keeps_backend_up(fake):
    backend.ensure_backend(fake.settings, 2)
    fake.set_busy(True)
    time.sleep(5)
    assert fake.running()
    fake.set_busy(False)
    assert fake.wait_until(lambda: not fake.running(), 15)


def _cold_call(port_env: dict, q) -> None:
    import os
    os.environ.update(port_env)
    info = backend.ensure_backend(config.load(), 60)
    q.put((info.started, info.supervisor_pid))


def test_concurrent_cold_calls_start_one_backend(fake):
    env = {k: v for k, v in __import__("os").environ.items() if k.startswith("LOCALGEN_")}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_cold_call, args=(env, q)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    results = [q.get(timeout=5) for _ in procs]
    assert sum(started for started, _ in results) == 1
    assert len({pid for _, pid in results}) == 1
    assert len(fake.launches()) == 1


def test_stop_backend_refuses_while_busy_unless_forced(fake):
    backend.ensure_backend(fake.settings)
    fake.set_busy(True)
    r = backend.stop_backend(fake.settings)
    assert not r["stopped"] and r["queue_length"] == 1
    r = backend.stop_backend(fake.settings, force=True)
    assert r["stopped"]
    assert not fake.running()
    assert backend.supervisor_pid(fake.settings) is None


def test_restart_after_stop(fake):
    backend.ensure_backend(fake.settings)
    assert backend.stop_backend(fake.settings)["stopped"]
    info = backend.ensure_backend(fake.settings, 90)
    assert info.started and info.idle_timeout_seconds == 90
    assert len(fake.launches()) == 2


def test_unmanaged_comfy_is_used_and_never_stopped(fake):
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(fake.port)],
                            cwd=fake.comfy_dir)
    try:
        assert fake.wait_until(fake.running, 10)
        info = backend.ensure_backend(fake.settings, 120)
        assert not info.managed and not info.started and info.idle_timeout_seconds is None
        r = backend.stop_backend(fake.settings)
        assert not r["stopped"] and r["running"]
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(10)
