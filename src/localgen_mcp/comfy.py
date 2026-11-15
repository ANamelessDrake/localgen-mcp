"""Minimal ComfyUI HTTP client (standard library only).

Ported from /ai/ComfyUI/zimage.py (the proven reference client): POST /prompt,
poll /history/<id>, fetch results through /view. Extended with image upload,
queue inspection and error reporting that raises instead of exiting.
"""

from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class ComfyError(RuntimeError):
    """ComfyUI rejected a request or a generation failed."""


class ComfyUnavailable(ComfyError):
    """ComfyUI is not reachable (connection refused, reset, timed out)."""


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # -- transport ---------------------------------------------------------

    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None,
                 timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:2000]
            raise ComfyError(f"ComfyUI {path} returned HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ComfyUnavailable(f"ComfyUI not reachable at {self.base_url}: {e}") from e

    def get_json(self, path: str, timeout: float | None = None) -> Any:
        return json.loads(self._request(path, timeout=timeout))

    def post_json(self, path: str, payload: Any) -> Any:
        raw = self._request(path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        return json.loads(raw) if raw.strip() else {}

    # -- API ---------------------------------------------------------------

    def alive(self, timeout: float = 2.0) -> bool:
        try:
            self.get_json("/system_stats", timeout=timeout)
            return True
        except ComfyError:
            return False

    def system_stats(self) -> dict:
        return self.get_json("/system_stats")

    def queue(self) -> dict:
        return self.get_json("/queue")

    def queue_length(self) -> int:
        q = self.queue()
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))

    def object_info(self, node_class: str | None = None) -> dict:
        return self.get_json("/object_info" + (f"/{node_class}" if node_class else ""), timeout=120)

    def submit(self, workflow: dict) -> str:
        resp = self.post_json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if "prompt_id" not in resp:
            raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(resp)[:2000]}")
        return resp["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        return self.get_json(f"/history/{prompt_id}").get(prompt_id)

    def upload_image(self, path: Path, subfolder: str = "localgen") -> str:
        """Upload a local file to ComfyUI's input folder; returns the name LoadImage expects.

        The stored name gets a random prefix, so concurrent jobs (several agents share one backend)
        never overwrite, or later delete, each other's inputs when their files share a name."""
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        stored = f"{uuid.uuid4().hex[:12]}_{path.name}"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{stored}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode() + path.read_bytes() + b"\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n{subfolder}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            f"--{boundary}--\r\n".encode(),
        ]
        raw = self._request("/upload/image", b"".join(parts),
                            {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=120)
        info = json.loads(raw)
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def download(self, item: dict, dest_dir: Path) -> Path:
        q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(item["filename"]).name
        dest.write_bytes(self._request(f"/view?{q}", timeout=600))
        return dest


def history_error(entry: dict) -> str | None:
    """Return a readable error message if a /history entry failed, else None."""
    status = entry.get("status", {})
    if status.get("status_str") != "error":
        return None
    for kind, msg in reversed(status.get("messages", [])):
        if kind == "execution_error":
            return f"{msg.get('node_type', '?')} (node {msg.get('node_id', '?')}): {msg.get('exception_message', msg)}"
    return json.dumps(status)[:2000]


def output_files(entry: dict) -> list[dict]:
    """Every saved file in a /history entry, whatever the node called its output list."""
    found = []
    for node_out in entry.get("outputs", {}).values():
        for value in node_out.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "filename" in item and item.get("type", "output") == "output":
                        found.append(item)
    return found
