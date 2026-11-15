#!/usr/bin/env bash
# Print the MCP client config block for localgen-mcp and refresh the copy in
# the repo (claude-mcp-config.json). It installs nothing: add the block to your
# MCP client's config yourself.
#
#   setup/register-mcp.sh [--root /ai] [--no-write]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ROOT="${LOCALGEN_ROOT:-/ai}"
WRITE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --no-write) WRITE=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

config="$(cat <<EOF
{
  "mcpServers": {
    "localgen": {
      "command": "$ROOT/venvs/localgen-mcp/bin/python",
      "args": ["-m", "localgen_mcp.server"],
      "env": {
        "LOCALGEN_ROOT": "$ROOT",
        "LOCALGEN_IDLE_TIMEOUT": "300"
      }
    }
  }
}
EOF
)"

printf '%s\n' "$config"
if (( WRITE )); then
    printf '%s\n' "$config" > "$REPO/claude-mcp-config.json"
    echo "(copy written to $REPO/claude-mcp-config.json)" >&2
fi
