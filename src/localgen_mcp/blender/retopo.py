"""Retopologize a textured GLB in Blender and bake its materials onto the result.

    blender -b --factory-startup -P retopo.py -- IN.glb OUT.glb FACES METHOD TEXTURE_SIZE [PYTHON]

METHOD is "quad" or "decimate". quad gives an all quad mesh with even edge
flow, which suits rigging, deformation and further modelling: Instant Meshes
through PYTHON (an interpreter with localgen_mcp and pynanoinstantmeshes),
else Blender's Quadriflow on a voxel remesh, else decimate. decimate collapses
to triangles and keeps the silhouette closest. The report says which ran.

The low poly mesh gets a fresh UV layout, and base color, roughness, metallic
and a tangent space normal map are baked from the original with Cycles on the
CPU (selected to active). Colour channels are baked as emission so no
lighting gets into them. The last stdout line is a JSON report starting with
"RETOPO_RESULT ".

Runs inside Blender's own Python (bpy); it is not imported by the MCP server.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector

t0 = time.time()
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
target_faces = int(argv[2])
method = argv[3]
tex_size = int(argv[4])
remesh_python = argv[5] if len(argv) > 5 else ""  # a Python with localgen_mcp and pynanoinstantmeshes
detail_source = argv[6] if len(argv) > 6 else "auto"  # auto | paint | none (decimate only)
detail_strength = float(argv[7]) if len(argv) > 7 else 1.0
report = {"input": src, "output": dst, "method_requested": method, "target_faces": target_faces}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in " + src)

# One high poly object, transforms applied.
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
high = bpy.context.view_layer.objects.active
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
high.name = "high"
report["input_faces"] = len(high.data.polygons)

lo = Vector((min(v.co[i] for v in high.data.vertices) for i in range(3)))
hi = Vector((max(v.co[i] for v in high.data.vertices) for i in range(3)))
diag = (hi - lo).length

def fresh_low():
    """A triangulated copy of the high poly mesh, without materials or UVs, made active."""
    old = bpy.data.objects.get("low")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    low = high.copy()
    low.data = high.data.copy()
    low.name = "low"
    bpy.context.scene.collection.objects.link(low)
    low.data.materials.clear()
    for uv in list(low.data.uv_layers):
        low.data.uv_layers.remove(uv)
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    tri = low.modifiers.new("tri", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=tri.name)
    return low


def weld(obj):
    """Merge vertices split along UV seams by the glTF import, so decimation sees one surface."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(hi - lo) * 1e-5)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def mesh_arrays(mesh):
    nv = len(mesh.vertices)
    normals = np.empty(nv * 3, dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", normals)
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    return normals.reshape(-1, 3), edges.reshape(-1, 2)


def smooth_field(values, edges, iterations=3):
    """Average each vertex with its neighbours, so the detail map has soft borders."""
    deg = np.bincount(edges.ravel(), minlength=len(values)).astype(np.float64)
    for _ in range(iterations):
        acc = values.astype(np.float64).copy()
        np.add.at(acc, edges[:, 0], values[edges[:, 1]])
        np.add.at(acc, edges[:, 1], values[edges[:, 0]])
        values = acc / (deg + 1.0)
    return values


def curvature_detail(mesh):
    """0..1 per vertex: how sharply the surface bends there (fingers, joints, face features, edges)."""
    normals, edges = mesh_arrays(mesh)
    bend = 1.0 - np.einsum("ij,ij->i", normals[edges[:, 0]], normals[edges[:, 1]])
    per_vertex = np.zeros(len(normals))
    np.maximum.at(per_vertex, edges[:, 0], bend)
    np.maximum.at(per_vertex, edges[:, 1], bend)
    per_vertex = smooth_field(per_vertex, edges)
    lo_q, hi_q = np.percentile(per_vertex, 50), np.percentile(per_vertex, 95)
    return np.clip((per_vertex - lo_q) / max(hi_q - lo_q, 1e-9), 0.0, 1.0)


def painted_detail(mesh):
    """0..1 per vertex from the mesh's colour attribute (white = full detail, black = least)."""
    attr = mesh.color_attributes.active_color or (mesh.color_attributes[0] if len(mesh.color_attributes) else None)
    if attr is None:
        raise SystemExit("detail='paint' needs vertex colours on the mesh (paint them in Blender, export a GLB)")
    raw = np.empty(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get("color", raw)
    lum = raw.reshape(-1, 4)[:, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if attr.domain == "CORNER":
        vidx = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", vidx)
        total = np.bincount(vidx, weights=lum, minlength=len(mesh.vertices))
        count = np.bincount(vidx, minlength=len(mesh.vertices))
        lum = total / np.maximum(count, 1)
    _, edges = mesh_arrays(mesh)
    return np.clip(smooth_field(lum, edges, 1), 0.0, 1.0)


def decimate_once(ratio, weights):
    low = fresh_low()
    if ratio < 1.0:
        mod = low.modifiers.new("dec", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        if weights is not None:
            # Blender collapses high-weight vertices first, so the group holds the detail map
            # and is inverted: high detail is kept, low detail goes first.
            group = low.vertex_groups.new(name="detail")
            for value in np.unique(np.round(weights, 2)):
                idx = np.nonzero(np.round(weights, 2) == value)[0].tolist()
                group.add(idx, float(value), "REPLACE")
            mod.vertex_group = group.name
            mod.invert_vertex_group = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return low


def decimate(faces):
    weld(high)  # once: every later copy starts welded
    weights = None
    tris = sum(len(p.vertices) - 2 for p in high.data.polygons)
    if detail_source != "none" and tris > faces:
        detail = curvature_detail(high.data) if detail_source == "auto" else painted_detail(high.data)
        # Strength 0 is uniform. The contrast is capped (weights stay within 0.15..0.85): with
        # weights near 0, Blender collapses those areas almost completely and the shape caves in.
        weights = 0.5 + 0.7 * detail_strength * (detail - 0.5)
        report["detail"] = {"source": detail_source, "strength": detail_strength,
                            "share_of_vertices_marked_detailed": round(float((detail > 0.5).mean()), 3)}
    if tris <= faces:
        return fresh_low()
    # A weighted collapse does not land exactly on the ratio; search for the one that hits the budget.
    ratio, best = faces / tris, None
    for _ in range(5):
        low = decimate_once(ratio, weights)
        n = len(low.data.polygons)
        if best is None or abs(n - faces) < abs(best[1] - faces):
            best = (ratio, n)
        if abs(n - faces) <= 0.03 * faces:
            break
        ratio = min(1.0, ratio * faces / max(n, 1))
    if len(low.data.polygons) != best[1]:
        low = decimate_once(best[0], weights)
    if weights is not None:
        # How the faces ended up: share of the result inside the areas marked detailed.
        report["detail"]["share_of_faces_in_detailed_areas"] = detail_face_share(low, weights)
    return low


def detail_face_share(low, weights):
    """Fraction of the low poly faces lying in the high detail areas of the original (nearest vertex)."""
    from mathutils.kdtree import KDTree
    tree = KDTree(len(high.data.vertices))
    for v in high.data.vertices:
        tree.insert(v.co, v.index)
    tree.balance()
    hits = sum(1 for p in low.data.polygons if weights[tree.find(p.center)[1]] > 0.5)
    return round(hits / max(len(low.data.polygons), 1), 3)


# Voxel grid sizes (cells across the largest dimension) to try for quad remeshing.
# Quadriflow needs a closed manifold, which a voxel remesh gives, but in Blender's
# build it also rejects fine voxel output of thin or near touching parts, so the
# grid gets coarser until it is accepted. Detail lost here comes back in the
# normal map, which is baked from the original.
QUAD_GRIDS = (300, 200, 140, 100, 70, 50, 35)

def instant_meshes(faces):
    """Quad remesh through the Instant Meshes helper process; None if it is unavailable or fails."""
    if not remesh_python:
        report["instant_meshes_error"] = "no helper python given"
        return None
    low = fresh_low()
    bm = bmesh.new()
    bm.from_mesh(low.data)
    # glTF import splits vertices along UV seams; weld them, or every seam is a hole.
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(hi - lo) * 1e-5)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.verts.index_update()
    verts = np.array([v.co[:] for v in bm.verts], dtype=np.float32)
    tris = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
    bm.free()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        src, dst = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(src, verts=verts, faces=tris)
        proc = subprocess.run([remesh_python, "-I", "-m", "localgen_mcp.instant_remesh", src, dst, str(faces)],
                              capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0 or not os.path.exists(dst):
            report["instant_meshes_error"] = (proc.stderr or proc.stdout)[-2000:]
            return None
        out = np.load(dst)
        qv, qf = out["verts"], out["faces"]
    polys = []
    for f in qf.tolist():
        uniq = list(dict.fromkeys(f))  # a repeated index marks a triangle
        if len(uniq) >= 3:
            polys.append(uniq)
    mesh = bpy.data.meshes.new("low")
    mesh.from_pydata(qv.tolist(), [], polys)
    mesh.validate()
    low.data = mesh
    return low


method_used = method
if method == "quad":
    low = instant_meshes(target_faces)
    if low is not None:
        method_used = "quad (Instant Meshes)"
    for cells in QUAD_GRIDS if low is None else ():
        cand = fresh_low()
        cand.data.remesh_voxel_size = max(hi - lo) / cells
        bpy.ops.object.voxel_remesh()
        before = len(cand.data.polygons)
        if bpy.ops.object.quadriflow_remesh(target_faces=target_faces, use_mesh_symmetry=False,
                                            use_preserve_sharp=False, seed=0) == {"FINISHED"} \
                and len(cand.data.polygons) != before:
            low = cand
            method_used = "quad (Quadriflow)"
            report["quad_voxel_cells"] = cells
            break
    if low is None:
        method_used = "decimate"
        report["note"] = "Quadriflow rejected this shape at every voxel size; fell back to decimate"
        low = decimate(target_faces)
elif method == "decimate":
    low = decimate(target_faces)
else:
    raise SystemExit("method must be quad or decimate")
report["method_used"] = method_used
report["output_faces"] = len(low.data.polygons)
report["output_vertices"] = len(low.data.vertices)
report["output_quads"] = sum(1 for p in low.data.polygons if len(p.vertices) == 4)

bpy.ops.object.shade_smooth()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.003)
bpy.ops.object.mode_set(mode="OBJECT")

# Target images and the low poly material.
# Colour values pass through the bake as raw bytes: every source texture is
# read as Non-Color (not decoded) and every target is Non-Color (not encoded),
# so the original's sRGB bytes come out as the same bytes on any Blender build,
# including ones whose OCIO config fails to load (Fedora's 5.2 package, which
# otherwise encoded twice and washed the colours out). Targets are never
# relabelled afterwards: changing a generated image's colour space regenerates
# it and wipes the bake. glTF reads base color as sRGB by definition.
for img in list(bpy.data.images):
    if img.users and img.source == "FILE":
        img.colorspace_settings.name = "Non-Color"


def image(name):
    img = bpy.data.images.new(name, tex_size, tex_size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color"
    return img

imgs = {"base_color": image("base_color"), "roughness": image("roughness"),
        "metallic": image("metallic"), "normal": image("normal")}
mat = bpy.data.materials.new("retopo")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
tex_nodes = {}
for i, (key, img) in enumerate(imgs.items()):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = img
    n.location = (-700, 300 - 300 * i)
    tex_nodes[key] = n
nt.links.new(tex_nodes["base_color"].outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(tex_nodes["roughness"].outputs["Color"], bsdf.inputs["Roughness"])
nt.links.new(tex_nodes["metallic"].outputs["Color"], bsdf.inputs["Metallic"])
nmap = nt.nodes.new("ShaderNodeNormalMap")
nmap.location = (-300, -600)
nt.links.new(tex_nodes["normal"].outputs["Color"], nmap.inputs["Color"])
nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
low.data.materials.append(mat)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 4
bake = scene.render.bake
bake.use_selected_to_active = True
# Rays start this far outside the low poly surface and search inward; the
# low poly mesh can sit well inside or outside thin parts of the original.
bake.cage_extrusion = diag * 0.03
bake.max_ray_distance = diag * 0.08
bake.margin = 8
bake.use_clear = True

bpy.ops.object.select_all(action="DESELECT")
high.select_set(True)
low.select_set(True)
bpy.context.view_layer.objects.active = low


def set_active_image(key):
    nt.nodes.active = tex_nodes[key]


def emission_rewire(socket_name):
    """Feed each high poly material's input socket into an emission shader so it bakes unlit."""
    saved = []
    for slot in high.material_slots:
        m = slot.material
        if m is None or not m.use_nodes:
            continue
        mt = m.node_tree
        out = next((n for n in mt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        p = next((n for n in mt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if out is None or p is None:
            continue
        old = [l.from_socket for l in out.inputs["Surface"].links]
        em = mt.nodes.new("ShaderNodeEmission")
        sock = p.inputs[socket_name]
        if sock.is_linked:
            mt.links.new(sock.links[0].from_socket, em.inputs["Color"])
        else:
            v = sock.default_value
            em.inputs["Color"].default_value = tuple(v) if hasattr(v, "__len__") else (v, v, v, 1.0)
        mt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        saved.append((mt, out, old, em))
    return saved


def emission_restore(saved):
    for mt, out, old, em in saved:
        mt.nodes.remove(em)
        for s in old:
            mt.links.new(s, out.inputs["Surface"])


timings = {}
for key, socket in (("base_color", "Base Color"), ("roughness", "Roughness"), ("metallic", "Metallic")):
    t = time.time()
    saved = emission_rewire(socket)
    set_active_image(key)
    bpy.ops.object.bake(type="EMIT")
    emission_restore(saved)
    timings[key] = round(time.time() - t, 1)
t = time.time()
set_active_image("normal")
bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
timings["normal"] = round(time.time() - t, 1)
report["bake_seconds"] = timings


def median_rgb(img, skip_black):
    px = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(-1, img.channels)[:, :3]
    if skip_black:  # texels outside the UV islands stay black
        rgb = rgb[rgb.sum(axis=1) > 0.03]
    return [round(float(v) * 255) for v in np.median(rgb, axis=0)] if len(rgb) else None


# Colour check: the baked base color should keep the original's values. A
# colour management slip shows up here as a large shift (washed out or dark).
sources = [l.from_node.image for s in high.material_slots if s.material and s.material.use_nodes
           for n in s.material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"
           for l in n.inputs["Base Color"].links if l.from_node.type == "TEX_IMAGE" and l.from_node.image]
report["base_color_median"] = {"source": median_rgb(sources[0], False) if sources else None,
                               "baked": median_rgb(imgs["base_color"], True)}

for img in imgs.values():
    img.pack()

bpy.ops.object.select_all(action="DESELECT")
low.select_set(True)
bpy.context.view_layer.objects.active = low
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True,
                          export_image_format="AUTO", export_apply=True)
# glTF stores triangles only, so the GLB holds each quad as two triangles. An
# FBX beside it keeps the real quads, with the textures embedded (Unity reads it).
fbx = os.path.splitext(dst)[0] + ".fbx"
bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, path_mode="COPY", embed_textures=True,
                         mesh_smooth_type="FACE", apply_scale_options="FBX_SCALE_ALL")
report["fbx"] = fbx
report["seconds"] = round(time.time() - t0, 1)
print("RETOPO_RESULT " + json.dumps(report), flush=True)
