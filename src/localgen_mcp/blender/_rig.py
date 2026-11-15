"""Pixal3D's fixed orbit rig in Blender (runs inside Blender's Python).

Shared by render_rig_views.py (generate_turnaround's mesh guide) and scripts/render_glb_views.py.
Cameras follow nodes_trellis2._orbit_camera_to_world: Z up, azimuth 0 looks along +Y (the camera
sits at -Y, in front of a generated mesh, which faces -Y after glTF import), azimuth 90 sits at +X
and sees the object's own left side, horizontal FOV 20 degrees, distance 1.1 * 0.5 / tan(10 deg),
so a unit object spans 1/1.1 of the frame. The mesh is scaled so its largest bounding box side is
1 and centred at the origin, as the rig assumes.
"""
import math

import bpy
from mathutils import Matrix, Vector

FOV_DEG = 20.0
DISTANCE = 1.1 * 0.5 / math.tan(math.radians(FOV_DEG / 2))
VIEWS = {"front": 0, "left": 90, "back": 180, "right": 270}


def frame(meshes):
    """Scale and centre the scene so the meshes' bounding box has its largest side 1, centred at the
    origin. Returns the bounding box size before scaling."""
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
    bpy.context.view_layer.update()
    return tuple(hi - lo)


def add_camera(scene):
    """The rig's camera, made the scene camera; place it with place(cam, azimuth)."""
    data = bpy.data.cameras.new("rig_cam")
    data.sensor_fit = "HORIZONTAL"
    data.angle = math.radians(FOV_DEG)
    cam = bpy.data.objects.new("rig_cam", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def place(cam, azimuth_deg):
    a = math.radians(azimuth_deg)
    back = Vector((math.sin(a), -math.cos(a), 0))
    right = Vector((math.cos(a), math.sin(a), 0))
    up = back.cross(right)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = right[i], up[i], back[i], back[i] * DISTANCE
    cam.matrix_world = m
