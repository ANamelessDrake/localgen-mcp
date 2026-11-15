"""Quad remesh a triangle mesh with Instant Meshes (pynanoinstantmeshes, BSD).

    python -m localgen_mcp.instant_remesh IN.npz OUT.npz TARGET_FACES

IN.npz holds "verts" (N x 3 float) and "faces" (M x 3 int) of a welded mesh;
OUT.npz gets the quad mesh in the same keys (faces K x 4). Run as a separate
process by blender/retopo.py, because Blender's own Python cannot load the
extension; the MCP server never imports this module or numpy.

Instant Meshes ends by splitting every face into quads, which multiplies the
face count by about four, so it is asked for a quarter of the target.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    src, dst, target = argv[0], argv[1], int(argv[2])
    import numpy as np
    import pynanoinstantmeshes as nim

    data = np.load(src)
    verts = np.ascontiguousarray(data["verts"], dtype=np.float32)
    faces = np.ascontiguousarray(data["faces"], dtype=np.uint32)
    out_v, out_f = nim.remesh(verts, faces, max(target // 4, 64), deterministic=True)
    np.savez(dst, verts=np.asarray(out_v, dtype=np.float32), faces=np.asarray(out_f, dtype=np.int64))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
