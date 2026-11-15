"""Render a GLB from Pixal3D's fixed orbit rig (front, left, back, right).

blender -b -t 8 -P scripts/render_glb_views.py -- in.glb out_prefix [size] [transparent 0|1] [wire 0|1] [object]
    [--pose 'REGEX=DEGREES' ...]

object: render only the mesh object of that name (for example one LOD of a LOD chain).

--pose rotates every bone whose name matches REGEX (re.fullmatch) about its
local X axis, rig_creature's bend axis, after framing the rest pose (so
stills of different poses share one scale): for example
--pose 'wing_.*_01=25' raises both wings of a 'wings' rig.

The mesh is scaled so its largest bounding box side is 1 and centred at the
origin, as the rig assumes; cameras follow nodes_trellis2._orbit_camera_to_world
(Z up, azimuth 0 looks along +Y, horizontal FOV 20, distance 1.1*0.5/tan(10 deg)).
The rig itself lives in src/localgen_mcp/blender/_rig.py, shared with the server's
render_rig_views.py (generate_turnaround's mesh guide).
"""
import math
import os
import re
import sys

import bpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "localgen_mcp", "blender"))
import _rig  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
poses = []
while "--pose" in argv:
    i = argv.index("--pose")
    pattern, deg = argv[i + 1].rsplit("=", 1)
    poses.append((pattern, float(deg)))
    del argv[i:i + 2]
src, prefix = argv[0], argv[1]
size = int(argv[2]) if len(argv) > 2 else 512
transparent = len(argv) > 3 and argv[3] == "1"
wire = len(argv) > 4 and argv[4] == "1"  # overlay the mesh edges
only = argv[5] if len(argv) > 5 else ""  # one object's name, for example "robot_LOD2"

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
# the glTF importer keeps a bone display shape (an icosphere) in a "glTF_not_exported" collection
for o in bpy.context.scene.objects:
    if any(c.name.startswith("glTF_not_exported") for c in o.users_collection):
        o.hide_render = True
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_render]
if only:
    for o in meshes:
        if o.name != only:
            bpy.data.objects.remove(o, do_unlink=True)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_render]
    if not meshes:
        raise SystemExit(f"no mesh object named {only!r} in {src}")

size3 = _rig.frame(meshes)
extent = max(size3)
print(f"bbox extent {tuple(round(v, 3) for v in size3)}, scaled by {1 / extent:.4f}")

# Pose after framing, so stills of different poses share one scale.
for rig in (o for o in bpy.context.scene.objects if o.type == "ARMATURE"):
    for pb in rig.pose.bones:
        for pattern, deg in poses:
            if re.fullmatch(pattern, pb.name):
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = (math.radians(deg), 0, 0)
bpy.context.view_layer.update()

if wire:
    # Draw edges with Cycles' Wireframe node mixed into each material (no extra geometry).
    for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
        mat.use_nodes = True
        nt = mat.node_tree
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
        surface = out.inputs["Surface"].links[0].from_socket
        mix = nt.nodes.new("ShaderNodeMixShader")
        edge = nt.nodes.new("ShaderNodeWireframe")
        edge.use_pixel_size = True
        edge.inputs["Size"].default_value = 1.0
        ink = nt.nodes.new("ShaderNodeBsdfDiffuse")
        ink.inputs["Color"].default_value = (0.01, 0.01, 0.01, 1)
        nt.links.new(edge.outputs["Fac"], mix.inputs["Fac"])
        nt.links.new(surface, mix.inputs[1])
        nt.links.new(ink.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 48
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.film_transparent = transparent
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.0
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(sun)

cam = _rig.add_camera(scene)
for name, az in _rig.VIEWS.items():
    _rig.place(cam, az)
    scene.render.filepath = f"{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True)
