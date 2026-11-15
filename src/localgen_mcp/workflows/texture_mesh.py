"""Texture an existing mesh: new UVs, painted views projected into a base color, and AO.

All native ComfyUI nodes:
  Load3D + Get3DComponents     read the uploaded GLB, OBJ or STL into a MESH
  UnwrapMesh                   fresh UV atlas (unless the mesh's own UVs are kept)
  MeshSmoothNormals            smooth shading for the clay renders
  CreateCameraInfo + Orbit     N cameras around the mesh, framed to its bounding sphere
  RenderMesh (solid)           grey clay views, with silhouette masks
  FLUX.2 Klein, per view       paints each clay view image to image (it starts from the
                               view's own latent with partial denoise, which pins the
                               layout to the geometry; from an empty latent Klein drifts
                               back to a front view). The front view may also take a
                               style reference image.
  BakeTextureFromViews         projects the painted views into the UV texture, each
                               limited to its silhouette, unseen texels filled
  BakeAmbientOcclusion         AO from the mesh itself
  ApplyTextureToMesh, SaveGLB

Normal (bump) and roughness maps are added afterwards in Blender (texture_maps.py).
"""

from __future__ import annotations

from . import Workflow
from .flux2_klein_4b import MODELS as KLEIN_MODELS, add_edit, add_loaders

VIEW_NAMES = {0: "front", 90: "side", 180: "back", 270: "side"}


def view_name(yaw: float) -> str:
    """The side of the object a camera at this orbit angle sees (0 is the front)."""
    nearest = min(VIEW_NAMES, key=lambda a: min(abs(yaw - a), 360 - abs(yaw - a)))
    off = min(abs(yaw - nearest), 360 - abs(yaw - nearest))
    return VIEW_NAMES[nearest] if off < 25 else f"three-quarter {VIEW_NAMES[nearest]}"


def view_prompt(name: str, prompt: str, styled: bool) -> str:
    style = " Match the colours, materials and style of image 2." if styled else ""
    return (f"Image 1 is a grey clay render of the {name} of a 3D model. Paint it as: {prompt}.{style} "
            f"This is the {name} view: show only what is visible from the {name}. Keep exactly the same shape, "
            "silhouette, pose, framing and camera as image 1; only add colour and surface materials. "
            "Even flat lighting, no shadows, plain white background.")


def build(p: dict) -> dict:
    n = int(p["views"])
    size = int(p["view_size"])
    g: dict = {
        "load": {"class_type": "Load3D", "inputs": {
            "model_file": p["mesh"], "width": size, "height": size,
            # Load3D's viewport state normally comes from the browser; headless it only
            # needs these keys and placeholder images.
            "image": {"image": p["placeholder"], "mask": p["placeholder"], "normal": p["placeholder"],
                      "recording": "", "camera_info": {}}}},
        "mesh": {"class_type": "Get3DComponents", "inputs": {"model_3d": ["load", 6]}},
    }
    mesh = ["mesh", 0]
    if p.get("unwrap", True):
        g["unwrap"] = {"class_type": "UnwrapMesh", "inputs": {
            "mesh": mesh, "segmenter": "pec", "resolution": int(p["texture_size"]), "padding": 2, "weld_distance": 0.0}}
        mesh = ["unwrap", 0]
    g["smooth"] = {"class_type": "MeshSmoothNormals", "inputs": {"mesh": mesh, "crease_angle": 60}}
    mesh = ["smooth", 0]
    g["cam"] = {"class_type": "CreateCameraInfo", "inputs": {
        "mode": "orbit", "mode.yaw": 0.0, "mode.pitch": float(p["pitch"]), "mode.distance": 1.8,
        "target_x": 0.0, "target_y": 0.0, "target_z": 0.0, "roll": 0.0, "fov": 35.0, "zoom": 1.0,
        "camera_type": "perspective", "mesh": mesh}}
    g["orbit"] = {"class_type": "CreateCameraOrbit", "inputs": {
        "camera_info": ["cam", 0], "count": n, "yaw": 360.0, "pitch": 0.0, "poles": "none"}}
    g["clay"] = {"class_type": "RenderMesh", "inputs": {
        "mesh": mesh, "mode": "solid", "width": size, "height": size, "background": "#ffffff",
        "camera_info": ["orbit", 0]}}

    add_loaders(g)
    style = p.get("images", {}).get("style")
    if style:
        g["style"] = {"class_type": "LoadImage", "inputs": {"image": style}}
    painted = []
    for v in range(n):
        g[f"view{v}"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["clay", 0], "batch_index": v, "length": 1}}
        refs = [[f"view{v}", 0]]
        styled = bool(style) and v == 0
        if styled:
            refs.append(["style", 0])
        text = view_prompt(view_name(360.0 * v / n), p["prompt"], styled)
        painted.append(add_edit(g, f"paint{v}_", text, refs, int(p["seed"]) + v, int(p["steps"]), size, size,
                                init=[f"view{v}", 0], denoise=float(p["denoise"]), upscale="lanczos"))
    batch = painted[0]
    for v in range(1, n):
        g[f"batch{v}"] = {"class_type": "ImageBatch", "inputs": {"image1": batch, "image2": painted[v]}}
        batch = [f"batch{v}", 0]

    g.update({
        "bake": {"class_type": "BakeTextureFromViews", "inputs": {
            "mesh": mesh, "images": batch, "camera_info": ["orbit", 0], "mask": ["clay", 1],
            "resolution": int(p["texture_size"]), "blend_exponent": 4.0, "fill_unseen": True}},
        "ao": {"class_type": "BakeAmbientOcclusion", "inputs": {
            "low_poly": mesh, "high_poly": mesh, "resolution": min(int(p["texture_size"]), 2048), "samples": 64,
            "max_distance": 0.5, "strength": 1.0, "bias": 0.01}},
        "apply": {"class_type": "ApplyTextureToMesh", "inputs": {"mesh": mesh, "base_color": ["bake", 0],
                                                                 "occlusion": ["ao", 0]}},
        "save_glb": {"class_type": "SaveGLB", "inputs": {"mesh": ["apply", 0], "filename_prefix": p["prefix"]}},
        "save_views": {"class_type": "SaveImage", "inputs": {"images": batch, "filename_prefix": p["prefix"] + "_view"}},
    })
    return g


WORKFLOW = Workflow(
    name="texture-klein",
    modality="texture",
    description=("Texture an existing mesh (GLB, OBJ or STL): fresh UVs, clay views painted by FLUX.2 Klein 4B "
                 "(Apache 2.0) from a prompt and optional style image, projected into a base color, plus AO"),
    models=list(KLEIN_MODELS),
    defaults={"views": 4, "pitch": 20.0, "denoise": 0.9, "steps": 6, "view_size": 1024, "texture_size": 2048,
              "unwrap": True},
    limits={"views": (2, 8), "steps": (2, 20), "view_size": (512, 1536), "texture_size": (512, 4096)},
    build=build,
    image_slots=("style",),
    is_default=True,
    needs_mesh=True,
)
