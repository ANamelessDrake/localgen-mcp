"""Render a GLB from Pixal3D's fixed orbit rig (front, left, back, right).

blender -b -t 8 -P scripts/render_glb_views.py -- in.glb out_prefix [size] [transparent 0|1] [wire 0|1]

The mesh is scaled so its largest bounding box side is 1 and centred at the
origin, as the rig assumes; cameras follow nodes_trellis2._orbit_camera_to_world
(Z up, azimuth 0 looks along +Y, horizontal FOV 20, distance 1.1*0.5/tan(10 deg)).
"""
import math
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, prefix = argv[0], argv[1]
size = int(argv[2]) if len(argv) > 2 else 512
transparent = len(argv) > 3 and argv[3] == "1"
wire = len(argv) > 4 and argv[4] == "1"  # overlay the mesh edges

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

lo = Vector((1e9,) * 3)
hi = Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector(map(min, lo, w))
        hi = Vector(map(max, hi, w))
centre, extent = (lo + hi) / 2, max(hi - lo)
fix = Matrix.Scale(1.0 / extent, 4) @ Matrix.Translation(-centre)
for o in bpy.context.scene.objects:
    if o.parent is None:
        o.matrix_world = fix @ o.matrix_world
print(f"bbox extent {tuple(round(v, 3) for v in (hi - lo))}, scaled by {1 / extent:.4f}")

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

cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.angle = math.radians(20)
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
dist = 1.1 * 0.5 / math.tan(math.radians(10))
for name, az in (("front", 0), ("left", 90), ("back", 180), ("right", 270)):
    a = math.radians(az)
    back = Vector((math.sin(a), -math.cos(a), 0))
    right = Vector((math.cos(a), math.sin(a), 0))
    up = back.cross(right)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = right[i], up[i], back[i], back[i] * dist
    cam.matrix_world = m
    scene.render.filepath = f"{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True)
