#!/usr/bin/env bash
# Download the model weights in setup/models.manifest that are missing.
#
#   setup/download-models.sh [--root /ai] [--only image|video|mesh] [--dry-run] [--verify]
#
# Weights go to <root>/ComfyUI/models. Caches and temp files stay under <root>.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LOCALGEN_ROOT:-/ai}"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

# A dry run changes nothing, not even these directories.
[[ " ${ARGS[*]-} " == *" --dry-run "* ]] || mkdir -p "$ROOT/tmp" "$ROOT/.cache/huggingface"
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/.cache/huggingface"

exec python3 "$HERE/download_models.py" --models-dir "$ROOT/ComfyUI/models" "${ARGS[@]}"
