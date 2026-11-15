#!/usr/bin/env python3
"""Download the model weights listed in setup/models.manifest.

Only files that are missing (or whose size is wrong) are downloaded. Each
download goes to a .part file next to its target (so nothing touches /tmp),
resumes if interrupted, and is checked against the manifest SHA256 before it
is moved into place.

Standard library only, so it runs before any venv exists.

    download_models.py --models-dir /ai/ComfyUI/models [--only video] [--dry-run] [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "models.manifest"
CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Entry:
    group: str
    target: str
    repo: str
    revision: str
    path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.path}"


def read_manifest(path: Path) -> list[Entry]:
    entries = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 7:
            raise SystemExit(f"{path}:{lineno}: expected 7 fields, got {len(parts)}")
        g, target, repo, rev, p, size, sha = parts
        entries.append(Entry(g, target, repo, rev, p, int(size), sha.lower()))
    return entries


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


def download(entry: Entry, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > entry.size:
        part.unlink()
        have = 0
    headers = {"User-Agent": "localgen-mcp-setup"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(entry.url, headers=headers)
    t0 = time.time()
    last = 0.0
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range; start over
        mode = "ab" if have else "wb"
        done = have
        with part.open(mode) as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 10:
                    rate = (done - have) / max(now - t0, 1e-6)
                    print(f"    {human(done)} / {human(entry.size)}  {human(rate)}/s", flush=True)
                    last = now
    size = part.stat().st_size
    if size != entry.size:
        raise RuntimeError(f"size mismatch for {entry.target}: got {size}, expected {entry.size} (rerun to resume)")
    print("    verifying sha256", flush=True)
    digest = sha256_file(part)
    if digest != entry.sha256:
        part.unlink()
        raise RuntimeError(f"sha256 mismatch for {entry.target}: got {digest}, expected {entry.sha256}")
    part.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", required=True, type=Path, help="ComfyUI models directory")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--only", action="append", choices=["image", "video", "mesh", "tts"],
                    help="limit to a group (repeatable); default all groups")
    ap.add_argument("--file", action="append", default=[], metavar="TARGET",
                    help="limit to manifest targets containing this text (repeatable), for example vae/ae")
    ap.add_argument("--dry-run", action="store_true", help="report what would be downloaded and exit")
    ap.add_argument("--verify", action="store_true", help="also checksum files that are already present")
    a = ap.parse_args()

    entries = [e for e in read_manifest(a.manifest)
               if (not a.only or e.group in a.only) and (not a.file or any(f in e.target for f in a.file))]
    missing: list[Entry] = []
    for e in entries:
        dest = a.models_dir / e.target
        if dest.exists() and dest.stat().st_size == e.size:
            if a.verify:
                ok = sha256_file(dest) == e.sha256
                print(f"[{'ok' if ok else 'BAD'}] {e.target}")
                if not ok:
                    missing.append(e)
            else:
                print(f"[present] {e.target}")
            continue
        missing.append(e)
        print(f"[missing] {e.target}  {human(e.size)}")

    total = sum(e.size for e in missing)
    print(f"{len(missing)} file(s) to download, {human(total)} in total, into {a.models_dir}")
    if a.dry_run or not missing:
        return 0

    a.models_dir.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(a.models_dir)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < total * 1.05:
        print(f"Not enough disk space: need {human(total)}, have {human(free_bytes)}", file=sys.stderr)
        return 1

    failed = 0
    for i, e in enumerate(missing, 1):
        dest = a.models_dir / e.target
        if dest.exists():
            dest.unlink()  # wrong size or bad checksum
        print(f"[{i}/{len(missing)}] {e.target} from {e.repo}", flush=True)
        try:
            download(e, dest)
            print(f"    done: {dest}", flush=True)
        except Exception as exc:  # report and continue with the next file
            failed += 1
            print(f"    FAILED: {exc}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
