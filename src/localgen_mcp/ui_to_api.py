"""Convert a ComfyUI UI workflow (the JSON the editor saves) to API format.

Development tool for adding workflows: ComfyUI ships its templates in UI
format, while /prompt needs API format. The editor's "Export (API)" does the
same thing; this lets it be done headless and reproducibly.

    python -m localgen_mcp.ui_to_api template.json --object-info http://127.0.0.1:8188 \
        [--switch 316=true] [--drop-class Preview3DAdvanced] > api.json

Handles widget values (including control_after_generate slots and nested
dynamic combos), links, bypassed and muted nodes, Note nodes, constant
switches, and prunes nodes that no output depends on. Subgraphs are not
supported (none of the workflows used here have them).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any

SKIP_TYPES = {"Note", "MarkdownNote", "Reroute"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COLOR"}
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"


def _is_widget(spec: list) -> bool:
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return False
    return isinstance(kind, list) or kind in WIDGET_TYPES or kind == DYNAMIC_COMBO


def _has_control(spec: list) -> bool:
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return bool(opts.get("control_after_generate"))


def _ordered_inputs(info: dict) -> list[tuple[str, list]]:
    order = info.get("input_order") or {}
    out = []
    for section in ("required", "optional"):
        specs = info.get("input", {}).get(section, {})
        names = order.get(section) or list(specs.keys())
        out.extend((n, specs[n]) for n in names if n in specs)
    return out


def _consume_widgets(inputs: list[tuple[str, list]], values: list, pos: int, prefix: str,
                     out: dict[str, Any]) -> int:
    for name, spec in inputs:
        if not _is_widget(spec):
            continue
        if pos >= len(values):
            break
        key = prefix + name
        value = values[pos]
        pos += 1
        out[key] = value
        if spec[0] == DYNAMIC_COMBO:
            options = spec[1].get("options", [])
            chosen = next((o for o in options if o.get("key") == value), None)
            if chosen is not None:
                sub = []
                for section in ("required", "optional"):
                    sub.extend(chosen.get("inputs", {}).get(section, {}).items())
                pos = _consume_widgets(sub, values, pos, key + ".", out)
        elif _has_control(spec):
            pos += 1  # the editor stores "fixed"/"randomize" after the value
    return pos


def convert(ui: dict, object_info: dict, switches: dict[int, bool] | None = None,
            drop_classes: set[str] | None = None, prune_unused: bool = True) -> dict:
    if ui.get("definitions", {}).get("subgraphs"):
        raise ValueError("subgraphs are not supported")
    switches = switches or {}
    drop_classes = drop_classes or set()
    links = {l[0]: (l[1], l[2]) for l in ui.get("links", [])}  # link id -> (origin node, slot)
    nodes = {n["id"]: n for n in ui["nodes"]}

    # Constant overrides for primitive nodes that drive switches (for example a PrimitiveBoolean).
    for nid, val in switches.items():
        nodes[nid]["widgets_values"] = [val] + list(nodes[nid].get("widgets_values", [])[1:])

    api: dict[str, dict] = {}
    for n in ui["nodes"]:
        if n["type"] in SKIP_TYPES or n.get("mode", 0) in (2, 4) or n["type"] in drop_classes:
            continue
        info = object_info.get(n["type"])
        if info is None:
            raise ValueError(f"node type {n['type']} (id {n['id']}) is not installed")
        inputs: dict[str, Any] = {}
        _consume_widgets(_ordered_inputs(info), list(n.get("widgets_values") or []), 0, "", inputs)
        for inp in n.get("inputs", []):
            if inp.get("link") is not None and inp["link"] in links:
                src, slot = links[inp["link"]]
                inputs[inp["name"]] = [str(src), slot]
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title") or n["type"]}}

    _resolve_constant_switches(api)
    return prune(api, object_info) if prune_unused else api


def _resolve_constant_switches(api: dict) -> None:
    """Rewire ComfySwitchNode whose switch is a constant (or a PrimitiveBoolean) to the chosen input."""
    changed = True
    while changed:
        changed = False
        for nid, node in list(api.items()):
            if node["class_type"] != "ComfySwitchNode":
                continue
            sw = node["inputs"].get("switch")
            if isinstance(sw, list):
                src = api.get(sw[0])
                if not src or src["class_type"] != "PrimitiveBoolean":
                    continue
                sw = src["inputs"].get("value")
            chosen = node["inputs"].get("on_true" if sw else "on_false")
            for other in api.values():
                for k, v in other["inputs"].items():
                    if v == [nid, 0]:
                        other["inputs"][k] = chosen
            del api[nid]
            changed = True


def prune(api: dict, object_info: dict) -> dict:
    """Keep only nodes that an output node depends on."""
    outputs = [nid for nid, n in api.items() if object_info.get(n["class_type"], {}).get("output_node")]
    keep: set[str] = set()
    stack = list(outputs)
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in api:
            continue
        keep.add(nid)
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {nid: api[nid] for nid in sorted(keep, key=lambda x: int(x) if x.isdigit() else x)}


def load_object_info(source: str) -> dict:
    if source.startswith("http"):
        with urllib.request.urlopen(source.rstrip("/") + "/object_info", timeout=120) as r:
            return json.load(r)
    with open(source) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow")
    ap.add_argument("--object-info", default="http://127.0.0.1:8188", help="ComfyUI URL or saved object_info JSON")
    ap.add_argument("--switch", action="append", default=[], help="NODE_ID=true|false for a PrimitiveBoolean")
    ap.add_argument("--drop-class", action="append", default=[], help="node class to remove (repeatable)")
    a = ap.parse_args()
    switches = {}
    for s in a.switch:
        k, _, v = s.partition("=")
        switches[int(k)] = v.lower() in ("1", "true", "yes")
    with open(a.workflow) as f:
        ui = json.load(f)
    api = convert(ui, load_object_info(a.object_info), switches, set(a.drop_class))
    json.dump(api, sys.stdout, indent=1, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
