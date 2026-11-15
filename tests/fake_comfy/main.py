"""Stand-in for ComfyUI's main.py, used by the backend tests.

Launched by the real supervisor exactly as ComfyUI would be
(`<python> main.py --listen HOST --port PORT --offline ...`, cwd = this folder's
copy). It answers the endpoints the supervisor and clients poll, records its
command line, and exits on SIGTERM like ComfyUI does.

Control files in the working directory:
    launches.log  one line per start: the JSON argv
    busy          when present, /queue reports one running job
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--listen", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8188)
ap.add_argument("--startup-delay", type=float, default=float(os.environ.get("FAKE_COMFY_STARTUP_DELAY", "0.5")))
args, _unknown = ap.parse_known_args()

with open("launches.log", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")

# Like ComfyUI, take a moment before the port answers.
time.sleep(args.startup_delay)


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/system_stats"):
            self._json({"system": {"comfyui_version": "fake"},
                        "devices": [{"name": "fake", "vram_total": 2**30, "vram_free": 2**29}]})
        elif self.path.startswith("/queue"):
            running = [[0, "busy-job"]] if os.path.exists("busy") else []
            self._json({"queue_running": running, "queue_pending": []})
        else:
            self.send_error(404)

    def log_message(self, *_a) -> None:
        pass


server = ThreadingHTTPServer((args.listen, args.port), Handler)
signal.signal(signal.SIGTERM, lambda *_: (server.server_close(), os._exit(0)))
server.serve_forever()
