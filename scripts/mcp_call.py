#!/usr/bin/env python3
"""Minimal MCP stdio client for smoke tests: start the server, list tools, call one.

    scripts/mcp_call.py --list
    scripts/mcp_call.py generate_image '{"prompt": "a red apple", "count": 1}'

Starts `python -m localgen_mcp.server` with this interpreter, exactly as an MCP
client would, and prints the tool result as JSON (progress messages go to stderr).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client


async def run(tool: str | None, args: dict, list_tools: bool) -> int:
    # The MCP client hands the server only a minimal environment (HOME, PATH, ...); pass PYTHONPATH and the
    # LOCALGEN_* settings through, so a checkout run with PYTHONPATH=src tests its own code and settings.
    env = {**get_default_environment(),
           **{k: v for k, v in os.environ.items() if k == "PYTHONPATH" or k.startswith("LOCALGEN_")}}
    params = StdioServerParameters(command=sys.executable, args=["-m", "localgen_mcp.server"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if list_tools:
                tools = await session.list_tools()
                for t in tools.tools:
                    print(t.name)
            if tool:
                t0 = time.time()

                async def on_progress(progress: float, total: float | None, message: str | None) -> None:
                    print(f"[progress {time.time() - t0:6.1f}s] {progress}/{total} {message}", file=sys.stderr, flush=True)

                result = await session.call_tool(tool, args, progress_callback=on_progress)
                for item in result.content:
                    text = getattr(item, "text", None)
                    print(text if text is not None else item)
                return 1 if result.isError else 0
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", nargs="?")
    ap.add_argument("arguments", nargs="?", default="{}")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    return asyncio.run(run(a.tool, json.loads(a.arguments), a.list))


if __name__ == "__main__":
    sys.exit(main())
