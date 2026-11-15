#!/usr/bin/env python3
"""Regenerate src/localgen_mcp/workflows/trellis2_api.json from ComfyUI's template.

Source: 3d_pixal3d_trellis2_image_to_model.json from the comfyui-workflow-templates
package, with its model switch (node 316) set to TRELLIS.2. Preview-only
branches are removed, preview passthrough nodes are bypassed, and the final
textured mesh (node 260) is saved with SaveGLB.

    scripts/make_trellis2_workflow.py --template PATH --object-info http://127.0.0.1:8188
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from localgen_mcp.ui_to_api import convert, load_object_info, prune

OUT = Path(__file__).resolve().parent.parent / "src" / "localgen_mcp" / "workflows" / "trellis2_api.json"
PREVIEW_ONLY = {"Preview3DAdvanced", "Save3DAdvanced"}
PASSTHROUGH = {"PreviewImage", "MaskPreview"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--object-info", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    info = load_object_info(a.object_info)
    ui = json.loads(Path(a.template).read_text())
    api = convert(ui, info, switches={316: True}, drop_classes=PREVIEW_ONLY, prune_unused=False)

    # Bypass preview nodes that the template uses as passthroughs.
    for nid, node in list(api.items()):
        if node["class_type"] in PASSTHROUGH:
            src = next(iter(node["inputs"].values()))
            for other in api.values():
                for k, v in other["inputs"].items():
                    if v == [nid, 0]:
                        other["inputs"][k] = src
            del api[nid]
    api["900"] = {"class_type": "SaveGLB", "inputs": {"mesh": ["260", 0], "filename_prefix": "localgen/mesh/x"},
                  "_meta": {"title": "Save textured GLB"}}
    api = prune(api, info)
    OUT.write_text(json.dumps(api, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(api)} nodes)")


if __name__ == "__main__":
    main()
