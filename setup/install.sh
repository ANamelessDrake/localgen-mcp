#!/usr/bin/env bash
# Rebuild the localgen-mcp stack on a machine with an NVIDIA GPU:
# ComfyUI at a pinned commit with its own venv, the MCP server venv, the run,
# log, output, temp and cache directories, and (unless skipped) the model weights.
#
#   setup/install.sh [--root /ai] [--cuda cu130] [--skip-models] [--only image|video|mesh ...] [--dry-run]
#
# Idempotent: every step checks what is already there and only does what is
# missing. --dry-run prints the plan and changes nothing.
# Everything large goes under the install root; nothing is written to /tmp or
# the home directory (uv's cache and its managed Pythons included).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"
COMFY_COMMIT="e08cc00304cd26879a23936886b947dd8a957966"   # ComfyUI 0.45.0
PYTHON_VERSION="3.12"
# Custom nodes: none are needed (every workflow uses native nodes). To add one,
# list "name git_url commit" lines in setup/custom_nodes.list.
CUSTOM_NODES_LIST="$HERE/custom_nodes.list"

ROOT="${LOCALGEN_ROOT:-/ai}"
CUDA=""
DRY=0
SKIP_MODELS=0
MODEL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --cuda) CUDA="$2"; shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --only) MODEL_ARGS+=(--only "$2"); shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n 2,13p "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
ROOT="$(realpath -m "$ROOT")"
COMFY_DIR="$ROOT/ComfyUI"
COMFY_VENV="$COMFY_DIR/.venv"
MCP_VENV="$ROOT/venvs/localgen-mcp"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export TMPDIR="$ROOT/tmp"

say() { printf '%s\n' "$*"; }
run() {
    if (( DRY )); then say "  would run: $*"; else say "  + $*"; "$@"; fi
}
step() { say ""; say "== $*"; }

# -- prerequisites ---------------------------------------------------------

step "Prerequisites"
missing=0
for cmd in nvidia-smi git uv; do
    if command -v "$cmd" >/dev/null; then say "  ok: $cmd"; else say "  MISSING: $cmd"; missing=1; fi
done
if command -v nvidia-smi >/dev/null; then
    gpu="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
    say "  GPU: $gpu"
    if [[ -z "$CUDA" ]]; then
        # Highest CUDA runtime the driver supports, mapped to a PyTorch wheel index.
        # Older drivers print "CUDA Version: 13.0", newer ones "CUDA UMD Version: 13.4".
        cuda_ver="$(nvidia-smi | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9]*\)\.\([0-9]*\).*/\2\3/p' | head -1)"
        say "  driver CUDA runtime: ${cuda_ver:-unknown}"
        if [[ -n "$cuda_ver" && "$cuda_ver" -ge 130 ]]; then CUDA=cu130
        elif [[ -n "$cuda_ver" && "$cuda_ver" -ge 128 ]]; then CUDA=cu128
        elif [[ -n "$cuda_ver" && "$cuda_ver" -ge 126 ]]; then CUDA=cu126
        else CUDA=cu130; say "  could not read the driver CUDA version; defaulting to $CUDA (override with --cuda)"; fi
    fi
fi
say "  PyTorch CUDA build: ${CUDA:-unknown}"
(( missing )) && { say "Install the missing tools and rerun."; exit 1; }

# -- directories -----------------------------------------------------------

step "Directories under $ROOT"
for d in "$ROOT/run/localgen" "$ROOT/logs" "$ROOT/output/localgen" "$ROOT/tmp" "$ROOT/.cache" "$ROOT/venvs"; do
    if [[ -d "$d" ]]; then say "  exists: $d"; else run mkdir -p "$d"; fi
done

# -- ComfyUI checkout ------------------------------------------------------

step "ComfyUI $COMFY_COMMIT"
if [[ -d "$COMFY_DIR/.git" ]]; then
    have="$(git -C "$COMFY_DIR" rev-parse HEAD)"
    if [[ "$have" == "$COMFY_COMMIT" ]]; then
        say "  ok: $COMFY_DIR at the pinned commit"
    else
        say "  $COMFY_DIR is at $have, not the pinned commit."
        say "  Not changing an existing checkout; to move it: git -C $COMFY_DIR fetch && git -C $COMFY_DIR checkout $COMFY_COMMIT"
    fi
else
    run git clone --filter=blob:none "$COMFY_REPO" "$COMFY_DIR"
    run git -C "$COMFY_DIR" checkout --detach "$COMFY_COMMIT"
fi

# -- ComfyUI venv ----------------------------------------------------------

step "ComfyUI venv ($COMFY_VENV)"
if [[ -x "$COMFY_VENV/bin/python" ]] && "$COMFY_VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    say "  ok: venv exists and torch sees CUDA ($("$COMFY_VENV/bin/python" -c 'import torch; print(torch.__version__)'))"
else
    [[ -x "$COMFY_VENV/bin/python" ]] || run uv venv --python "$PYTHON_VERSION" "$COMFY_VENV"
    # The lock pins torch to +cu130 builds; for another CUDA build, install torch
    # from that index first and the rest of the lock without the torch lines.
    if [[ "$CUDA" == "cu130" ]]; then
        run uv pip install --python "$COMFY_VENV/bin/python" \
            --index-strategy unsafe-best-match --extra-index-url "https://download.pytorch.org/whl/$CUDA" \
            -r "$HERE/comfyui-requirements.lock"
    else
        run uv pip install --python "$COMFY_VENV/bin/python" --index-url "https://download.pytorch.org/whl/$CUDA" \
            torch torchvision torchaudio
        if (( DRY )); then say "  would install setup/comfyui-requirements.lock without the torch lines"
        else
            grep -vE '^(torch|torchvision|torchaudio)==' "$HERE/comfyui-requirements.lock" > "$TMPDIR/comfyui-requirements.notorch"
            run uv pip install --python "$COMFY_VENV/bin/python" -r "$TMPDIR/comfyui-requirements.notorch"
        fi
    fi
fi

# -- custom nodes ----------------------------------------------------------

step "Custom nodes"
if [[ -s "$CUSTOM_NODES_LIST" ]]; then
    while read -r name url commit; do
        [[ -z "${name:-}" || "$name" == \#* ]] && continue
        dest="$COMFY_DIR/custom_nodes/$name"
        if [[ -d "$dest/.git" ]] && [[ "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]]; then
            say "  ok: $name at $commit"; continue
        fi
        [[ -d "$dest/.git" ]] || run git clone --filter=blob:none "$url" "$dest"
        run git -C "$dest" checkout --detach "$commit"
        [[ -f "$dest/requirements.txt" ]] && run uv pip install --python "$COMFY_VENV/bin/python" -r "$dest/requirements.txt"
    done < "$CUSTOM_NODES_LIST"
else
    say "  none required (all workflows use native ComfyUI nodes; ComfyUI-Manager is deliberately not installed)"
fi

# -- MCP server venv -------------------------------------------------------

step "MCP server venv ($MCP_VENV)"
if [[ -x "$MCP_VENV/bin/python" ]] && "$MCP_VENV/bin/python" -c "import localgen_mcp, mcp" 2>/dev/null; then
    say "  ok: venv exists and imports localgen_mcp from $("$MCP_VENV/bin/python" -c 'import localgen_mcp, os; print(os.path.dirname(localgen_mcp.__file__))')"
else
    [[ -x "$MCP_VENV/bin/python" ]] || run uv venv --python "$PYTHON_VERSION" "$MCP_VENV"
    run uv pip install --python "$MCP_VENV/bin/python" -r "$HERE/mcp-requirements.lock"
    run uv pip install --python "$MCP_VENV/bin/python" --no-deps -e "$REPO"
fi

# -- models ----------------------------------------------------------------

step "Model weights"
if (( SKIP_MODELS )); then
    say "  skipped (--skip-models); run setup/download-models.sh --root $ROOT later"
elif (( DRY )); then
    "$HERE/download-models.sh" --root "$ROOT" --dry-run "${MODEL_ARGS[@]}" | sed 's/^/  /'
else
    run "$HERE/download-models.sh" --root "$ROOT" "${MODEL_ARGS[@]}"
fi

step "Done"
say "  MCP client config: setup/register-mcp.sh --root $ROOT"
(( DRY )) && say "  (dry run: nothing was changed)"
exit 0
