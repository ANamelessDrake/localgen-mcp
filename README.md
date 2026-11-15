# localgen-mcp

A local MCP server that lets agents generate images, video and 3D meshes on this machine's GPU, fully offline, through one shared [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend that starts on demand and shuts itself down when idle.

- **Localhost and offline only.** The managed ComfyUI binds `127.0.0.1` and always runs with `--offline` (no partner or API nodes, a restrictive CSP for its frontend). Hub lookups are disabled with `HF_HUB_OFFLINE=1`. The network is used only by the setup scripts, to fetch software and weights.
- **Nothing resident when idle.** The MCP server imports no ML libraries. ComfyUI runs only while there is work, and its process is killed after the idle timeout (default 300 s), which releases all VRAM and RAM.
- **One backend for every agent.** Each agent session runs its own MCP server process; they all share a single ComfyUI instance, coordinated with file locks.
- **Returns file paths**, plus the seed and settings used, so results can be reproduced.

## Tools

| Tool | What it does |
|---|---|
| `generate_image` | Text to image (Z-Image Turbo). With `reference_images` (up to 4 paths), edits or combines them as the prompt says (FLUX.2 Klein 4B). `count` queues several at once with one model load. |
| `generate_video` | Text to video, or image to video with `input_image` as the first frame (Wan 2.2 TI2V 5B, MP4). |
| `generate_mesh` | Textured GLB from `input_image` (TRELLIS.2), or from `prompt` alone (an image is generated first). With `left_image`, `back_image` and/or `right_image`, builds from several views (Pixal3D multi-view); `framing` picks how the views are read (below). |
| `retopologize_mesh` | Game weight copy of any textured GLB: quad remesh (Instant Meshes) or decimation to `target_faces`, new UVs, and base color, roughness, metallic and normal maps baked from the original. Writes a GLB and an FBX (the FBX keeps real quads). `lods` adds a LOD chain sharing one texture set. Headless Blender on the CPU, about 10 to 30 s. `generate_mesh` runs it too when given `retopo_faces` or `retopo_lods`. |
| `texture_mesh` | Texture an existing mesh (GLB, glTF, OBJ or STL) from a prompt and optional `style_image`: fresh UVs, a painted base color projected from several views, AO, a bump-derived normal map, roughness and metallic. Writes a GLB, an FBX and the painted views. About 35 s on the GPU plus a few seconds in Blender. |
| `generate_parts` | Build a model as separate part meshes from a front view concept and a box for each part, assembled in place and rigidly rigged (one bone per part, pivoting at the joints). Writes a GLB and an FBX. About 1 to 2 minutes per part. |
| `generate_material` | Tileable PBR surface material from a prompt (optional `reference_image`, or `tile_image` to make an existing picture tile): a seamless base color plus height, normal (OpenGL +Y, or DirectX), roughness and AO maps, a 3x3 tile preview and a lit Cycles render. 1024 or 2048 px; about 25 s warm at 1024 plus 5 to 15 s in Blender. |
| `status` | Whether the backend is running and managed, the effective idle timeout, queue length, GPU memory, and which models are installed. |
| `start_backend` | Warm the backend ahead of a batch, optionally with a longer `idle_timeout_seconds`. |
| `stop_backend` | Stop it now to free memory. Refuses while jobs are queued unless `force=true`. |

Every generation tool takes optional `output_dir`, `seed`, `model` and `idle_timeout_seconds`. Outputs default to `/ai/output/localgen/<modality>/<date>/`. Long jobs send MCP progress notifications.

### Multi-view framing

Pixal3D assumes a fixed camera rig: views 90 degrees apart, the object centred and at the same scale in every view.
- `framing="crop"` (default, as in ComfyUI's template) crops each view to its own outline. This keeps the scale when the object's widest extent is the same in every view, as for a standing figure or a character turnaround sheet. For an object much wider from the front than the side it stretches the side views, and the mesh comes out too deep.
- `framing="as_framed"` uses square views exactly as given. Use it for 3D renders or turnarounds already framed like the rig (object about 1/1.1 of the frame at its widest, 20 degree horizontal FOV); `scripts/render_glb_views.py` renders that framing.
- `remove_background=false` takes the mask from the images' own alpha channel instead of BiRefNet.

### Retopology

Generated meshes are about 300k faces with textures baked for that mesh. For a game asset:
- `generate_mesh(face_count=...)` (5000 or more) decimates inside the generation and bakes the textures straight from the model's voxels: the best texture quality, but triangles.
- `retopologize_mesh` (or `generate_mesh(retopo_faces=...)`) rebuilds the topology of any GLB.
  - `method="quad"` (default) gives an even all-quad mesh, suited to rigging and deformation.
  - `method="decimate"` gives triangles that keep the silhouette closest.
  - It then unwraps new UVs and bakes base color, roughness, metallic and a tangent space normal map from the original with Cycles on the CPU. Colour values are copied byte for byte (a test checks this).
  - **Where the detail goes** (`method="decimate"` only):
    - `detail="auto"` (default) keeps more faces where the surface bends sharply, such as fingers, joints and face features, and fewer on flat panels.
    - `detail="paint"` reads the mesh's vertex colours as a map you paint in Blender: white for the most detail, black for the least. Export as GLB, which keeps vertex colours.
    - `detail="none"` is uniform. `detail_strength` (0 to 1) scales the effect; 1 is the most that keeps the shape intact.
    - The report gives the share of faces that landed in the detailed areas; on the test robot, the 16% of the surface marked detailed got 27% of the faces.
    - The quad remeshers have no per-area density, so quad output is uniform.
  - Thin parts far from the new surface can bake with dark specks; a higher `target_faces` reduces them.
  - glTF stores only triangles, so the GLB holds each quad as two triangles; the FBX written beside it keeps the quads and embeds the textures.
  - **LOD chains:** `lods=[10000, 3000, 800]` (or `generate_mesh(retopo_lods=...)`) writes LOD0, LOD1, LOD2 that share one UV layout and one texture set, so the engine loads one material.
    - `lods[0]` replaces `target_faces`: LOD0 is the retopology above, with its baked textures.
    - Each further LOD is a collapse decimation of LOD0 itself (not of the LOD before it), to that many triangles. Blender's collapse interpolates the UVs on each side of a UV seam separately, so LOD0's layout and textures still fit.
    - One GLB and one FBX hold objects `<name>_LOD0`, `<name>_LOD1`, ...; Unity's FBX importer builds a LOD Group from that suffix. The FBX keeps LOD0's quads; the lower LODs are triangles.
    - The report lists each LOD's face and triangle counts and a `texture_fit`, measured at each triangle's centre against the nearest point of LOD0: `uv_drift_texels_median` (a few texels when the layout is shared; a fresh unwrap measures about 100) and `colour_mismatch_share` (triangles whose base colour differs from LOD0's there, which grows as small parts collapse). The p90 drift can be large on a correct LOD, because a centre near a seam can find its nearest LOD0 point on the other island.
    - Detail weighting: with `method="decimate"`, the same detail map (`detail`, `detail_strength`) also steers the LOD decimations, carried from the original to LOD0's vertices. Without it, LODs made from a weighted LOD0 drift back toward uniform density (on the painted test mesh, 0.78 and 0.63 of the faces in the detailed half instead of 0.91 and 0.81). With `method="quad"` the LODs are uniform, like LOD0.
    - Limits: the lower LODs reuse LOD0's tangent space normal map, which fits less well as the shape simplifies. Strong reductions show it: on a test robot, a quad LOD0 of 18.8k triangles held up well down to 800 triangles, while a decimated LOD0 of 8k triangles taken to 600 grew spikes on thin parts (a direct 600 face decimation of the original has them too, though fewer). Quad LOD0s (the default) chain better.

Quad remeshing uses [Instant Meshes](https://github.com/wjakob/instant-meshes) (BSD) through `pynanoinstantmeshes` in the MCP venv. Blender's own Quadriflow is the fallback but rejects many generated meshes, so expect `decimate` results if that package is missing. Blender (4.2 or later, any build) must be installed; `LOCALGEN_BLENDER` points at a specific one.

### Texturing

`texture_mesh` gives any mesh a PBR material, using native ComfyUI nodes and FLUX.2 Klein:
1. **UVs:** `UnwrapMesh` makes a fresh UV layout; `keep_uvs=True` keeps the mesh's own.
2. **Clay renders:** `RenderMesh` draws grey clay views from `views` cameras around the mesh, framed to its bounding sphere and tilted 20 degrees.
3. **Painting:** Klein paints each view image to image. It starts from the clay render's own latent with partial `denoise`, which holds the painting to the geometry; from an empty latent it drifts back to a front view, and the back of the mesh gets a face. `style_image` guides the front view.
4. **Projection:** `BakeTextureFromViews` projects the painted views into the UVs, each limited to its silhouette, and fills texels no view sees. `BakeAmbientOcclusion` adds AO.
5. **Blender (CPU):** adds a tangent space normal map baked from a bump of the painted detail, a roughness map (rougher where the paint is darker) and metallic, then writes the GLB and an FBX.

Inputs are GLB, glTF, OBJ, STL, PLY or FBX. Each is re-exported through Blender, and only that fresh GLB is uploaded: like images, which must be real image files, a caller's file is never copied into ComfyUI's input folder as is. Uploads are deleted when the job finishes.

Limits: every view is painted separately, so the views can disagree on details. Parts seen only edge-on, or only from above, get filled or blended colour. It works best on clean meshes of a few thousand to 50k faces, such as `retopologize_mesh` output.

### Separate parts and a rigid rig

`generate_parts` is for models that animate as hinged parts (robots, machines, armoured creatures).
- **Input:** a front view concept image and, for each part, a name and a rough box (fractions of the image or pixels). The calling agent can read the boxes off the image. An optional parent and a few words of description can be given.
- **Isolation:** each box is cropped, and FLUX.2 Klein redraws just that part as a complete standalone piece, with hidden areas filled in. Asking Klein to isolate or recolour a part of the whole image was unreliable (it moved, merged or missed parts), so the crop matters.
- **Meshing:** TRELLIS.2 meshes each part (`face_count`, default 20k).
- **Placement:** each part is placed by its box: scaled to fit it, with depth taken from a whole-object mesh made from the same concept, which is also given back as `full_mesh`.
- **Rig:** one bone per part. The hierarchy follows the given parents, else which parts touch (a minimum spanning tree rooted at the largest part). Each bone pivots where its part meets its parent, and each part mesh is parented rigidly to its bone. The output is a GLB and an FBX with every part as its own object.
- **Resume:** a failed run (for example the GPU running out of memory while another application holds VRAM) resumes when called again with the same `output_dir`, reusing finished parts.

### Tileable materials

`generate_material` makes surface materials for environment art (rock, hull plating, floors, organic growth).
- **Base color:** Z-Image Turbo from the prompt, with "seamless top-down texture" wording added; FLUX.2 Klein when `reference_image` is given; or the caller's `tile_image`.
- **Making it tile:** image models do not tile. The graph (`workflows/seamless_material.py`, native nodes only) offsets the image by half its size with wrap around, so the old edges meet in a cross through the middle and the new edges are continuous. It then repaints over that cross in two passes with Z-Image image to image, each with its mask as the latent noise mask (`SetLatentNoiseMask`), compositing only the masked part back:
  1. A cross shaped band (`seam_width`, default 1/4 of the size across), fully repainted over the old edges and fading out towards its own edge, which noise makes ragged: a band with straight edges showed up as a grid once tiled. Its arms stop short of the image edges, because repainting up to an edge paints the two sides of that edge independently and breaks the wrap.
  2. What is left of the seam (short pieces where the arms stopped, two spots on the torus) is moved inside the image by a further quarter offset, repainted with a patch each, and moved back.
  No pass paints up to the edge of the frame it paints in, so the result wraps whatever the model draws (a test runs the graph with a stand-in that draws an image that does not tile). `seam_denoise` (default 0.7) sets how freely the bands are repainted; higher hides a stubborn seam but drifts further from the surrounding texture.
- **Seam check:** each wrap edge is compared with the opposite edge, and the repainted centre lines with their neighbours. A seam passes when its mean difference is no larger than the largest difference between neighbouring rows or columns inside the image (the result also gives its ratio to the mean and its percentile rank). If the centre still shows a line, the band is repainted again on the result (`seam_passes`, default 2 in all), and the new pass is kept only if its centre line is weaker. The result reports every metric, for the untiled image too.
- **Maps** (Blender's numpy on the CPU, every filter wraps around, so the maps tile as well):
  - height (16 bit PNG): luminance, high pass (a blur of 1/8 the size removed), light blur. Brighter reads as higher; `invert_height=true` for the reverse.
  - normal: tangent space from the height with wrap-around central differences, `normal_strength`. OpenGL convention by default (green is +Y, as Unity and Blender expect): `_normal_gl.png`. `directx_normal=true` writes DirectX -Y (Unreal): `_normal_dx.png`.
  - roughness: `roughness` as the base, rougher in cavities and darker spots and smoother on raised, bright areas by up to `roughness_variation`.
  - ambient occlusion: from the cavities of the height at three scales, `ao_strength` deep at most.
- **Previews:** `_tile3x3.png` (the base color 3x3) and `_preview.png`, a Cycles render of a sphere (or `preview="plane"`) with every map under a low, raking light.
- Files go to `<output_dir>/<name>_<seed>/` (default `/ai/output/localgen/material/<date>/`).

Limits: the maps are derived from the colour, so relief follows brightness, not real geometry: painted dark stains read as dents. The roughness is a heuristic, and there is no metallic map. The repaint can soften or change detail inside the band, which can show as a faint softer strip through the middle when tiled; look at the 3x3 preview. The seam metric is conservative for grids: on panel plating the wrap edge falls on a panel line, which can be the strongest transition in the image and then reads as not ok although it is not a seam.

### Idle timeout

- Default 300 s, bounded to 60 to 7200 s (`LOCALGEN_IDLE_TIMEOUT`, `LOCALGEN_IDLE_TIMEOUT_MIN`, `LOCALGEN_IDLE_TIMEOUT_MAX`).
- The call that starts the backend sets the timeout. While it runs, a longer request extends it and a shorter one is ignored: the effective value is the maximum requested since it started, so one agent cannot cut short another's batch.
- The effective value is in every tool result and in `status`.
- Idle means an empty ComfyUI queue and no client activity for that long.

## Models

Chosen for a 12 GB card (an RTX 3080 Ti, with about 7.8 GB actually free beside a desktop session). Reasons, rejected candidates and sources are in [docs/model-choices.md](docs/model-choices.md).

| Use | Model | License | Download |
|---|---|---|---|
| Text to image | Z-Image Turbo 6B bf16 | Apache 2.0 | 18.28 GB (with its text encoder and VAE) |
| Image editing, reference images; painting views for `texture_mesh` | FLUX.2 Klein 4B fp8 | Apache 2.0 | 12.45 GB (includes the full Qwen3 4B text encoder) |
| Video | Wan 2.2 TI2V 5B fp16 | Apache 2.0 | 18.15 GB |
| Mesh from one image | TRELLIS.2 4B int8 | MIT | 8.96 GB |
| Mesh from several views | Pixal3D multi-view int8 | MIT | 5.58 GB (shares the TRELLIS.2 files) |

All of them use native ComfyUI nodes; no custom nodes are installed.

### Measured on an RTX 3080 Ti (2026-11-15)

Measured through the MCP server over stdio, with all models in one backend session. Peak VRAM is the whole card: about 3.6 GB of it is the desktop and other applications. ComfyUI streams weights to fit, and no run hit an out of memory error.

| Job | Settings | Time (execution, total) | Peak VRAM |
|---|---|---|---|
| Image, cold start (process start + model load) | Z-Image Turbo 1024x1024, 8 steps | 66 s, 73 s | 11.5 GB |
| Image burst, `count=4`, warm | same | 8.4, 7.6, 7.7, 7.6 s per image; 33 s total | 11.8 GB |
| Image edit, one reference, first load | Klein 4B, 4 steps, 1024x1024 | 42 s | 11.7 GB |
| Image edit, warm | same | 6 s | 11.5 GB |
| Video, text to video | Wan 2.2 5B, 832x480, 49 frames (2.0 s at 24 fps), 20 steps | 85 s | 11.8 GB |
| Video, defaults, cold start | Wan 2.2 5B, 1280x704, 49 frames, 20 steps | 168 s, 174 s | 11.8 GB |
| Video, image to video, warm | same, 33 frames | 30 s | 11.8 GB |
| Mesh, one image | TRELLIS.2, resolution 1024, 2048 px textures, 300k faces | 76 s (GLB 16.9 MB, 162k vertices) | 11.5 GB |
| Mesh, four views | Pixal3D multi-view, same settings | 63 to 83 s (GLB about 17 MB) | 11.1 GB |
| Mesh, three views from a turnaround sheet | Pixal3D multi-view, `framing="crop"` | 74 to 96 s | |
| Retopology | 300k faces to 10k quads, 2048 px textures | 10 to 13 s (CPU) | none |
| Texture a mesh, warm | 4 views at 1024 px, 2048 px textures, 10k quad mesh | 34 s, plus 4 s in Blender | |
| Idle shutdown | default 300 s | ComfyUI gone at 301 s idle; VRAM back to 3.6 GB, RAM freed (about 18 GB) | |

Times include the model swap when the previous job used a different model.

Mesh quality was checked by rendering each GLB from the four rig views in Blender (`scripts/render_glb_views.py`). The test was Pixal3D fed four renders of a TRELLIS.2 teapot. With `framing="as_framed"` it reproduced the original's proportions: a bounding box of 0.993 x 0.609 x 0.654 against 1.0 x 0.614 x 0.654, with one handle and one spout. With `crop` the same views produced a body as wide from the side as from the front.

## Install or rebuild

Requirements: Linux, an NVIDIA GPU and driver, `git` and [`uv`](https://docs.astral.sh/uv/). uv provides Python 3.12. Blender is optional: only retopology needs it, and `install.sh` reports whether it was found.

```bash
setup/install.sh --root /ai            # ComfyUI, both venvs, directories, all model weights
setup/install.sh --root /ai --dry-run  # show what would be done, change nothing
setup/install.sh --skip-models         # software only
setup/download-models.sh --only video  # weights for one group (image, video, mesh); --verify rechecks SHA256
```

What gets pinned:
- ComfyUI at commit `e08cc003` (0.45.0).
- Both venvs from frozen lock files (`setup/comfyui-requirements.lock`, `setup/mcp-requirements.lock`).
- Every weight file by Hugging Face revision and SHA256 (`setup/models.manifest`).

Everything large lives under the install root: weights, venvs, uv's cache and managed Pythons, logs, outputs and temp files. Nothing goes in `/tmp` or the home directory. The scripts are idempotent; rerun them after an interruption.

### MCP client config

`setup/register-mcp.sh` prints the block below and keeps a copy in [claude-mcp-config.json](claude-mcp-config.json). It does not install it anywhere; add it to your MCP client's config yourself.

```json
{
  "mcpServers": {
    "localgen": {
      "command": "/ai/venvs/localgen-mcp/bin/python",
      "args": ["-m", "localgen_mcp.server"],
      "env": {
        "LOCALGEN_ROOT": "/ai",
        "LOCALGEN_IDLE_TIMEOUT": "300"
      }
    }
  }
}
```

## How it works

```
agent A ── MCP server ─┐                                   ┌─ ComfyUI 127.0.0.1:8188 --offline
agent B ── MCP server ─┼─ flock start.lock ─ supervisor ───┤   (child process, killed when idle)
agent C ── MCP server ─┘   touch last_activity             └─ reads idle_timeout every poll
```

- `backend.ensure_backend()` takes an exclusive lock on `<root>/run/localgen/start.lock`. If ComfyUI answers, it is used. Otherwise a detached supervisor is started (its own session, logs in `<root>/logs`), and the caller waits for `/system_stats`. Concurrent cold calls therefore start exactly one backend.
- The supervisor holds `supervisor.lock` for its lifetime and runs ComfyUI as its child, which gets SIGTERM if the supervisor dies. It polls `/queue` and the mtime of `last_activity`, which clients touch before and during every job.
- To shut down, the supervisor takes `start.lock` (so no client can submit into a dying server) and rechecks for activity. It then terminates ComfyUI (SIGTERM, then SIGKILL after a grace period) and clears its run state.
- If a ComfyUI that localgen did not start is already on the port (for example one started by hand), it is used as is and never stopped.

Run state, all under `<root>/run/localgen`: `start.lock`, `supervisor.lock`, `supervisor.pid`, `last_activity`, `idle_timeout`.

## Configuration

| Variable | Default | |
|---|---|---|
| `LOCALGEN_ROOT` | `/ai` | Install root |
| `LOCALGEN_COMFY_DIR` | `<root>/ComfyUI` | |
| `LOCALGEN_HOST`, `LOCALGEN_PORT` | `127.0.0.1`, `8188` | Non-loopback hosts are rejected |
| `LOCALGEN_IDLE_TIMEOUT` (`_MIN`, `_MAX`) | 300 (60, 7200) s | |
| `LOCALGEN_START_TIMEOUT` | 180 s | Wait for ComfyUI to answer |
| `LOCALGEN_TIMEOUT_IMAGE`, `_VIDEO`, `_MESH` | 900, 5400, 3600 s | Per job |
| `LOCALGEN_COMFY_ARGS` | | Extra ComfyUI flags, for example `--lowvram` |
| `LOCALGEN_OUTPUT_DIR`, `LOCALGEN_LOG_DIR`, `LOCALGEN_RUN_DIR` | under the root | |
| `LOCALGEN_MCP_LOG_LEVEL`, `LOCALGEN_MCP_LOG_FILE` | `INFO` | Server logs go to stderr |
| `LOCALGEN_BLENDER` | `blender` on `PATH` | For retopology |
| `LOCALGEN_BLENDER_THREADS` | 8 | CPU threads for Blender's bake |
| `LOCALGEN_TIMEOUT_RETOPO` | 1800 s | |

## Adding a model

1. Add its files to `setup/models.manifest`: group, target path, Hugging Face repo, pinned revision, path in the repo, size and SHA256 (all from `https://huggingface.co/api/models/<repo>/revision/<rev>?blobs=true`).
2. Add a module under `src/localgen_mcp/workflows/` defining `WORKFLOW = Workflow(...)`: modality, model files, defaults, limits, named `image_slots`, and a `build(params)` function returning an API format graph. List it in `_MODULES`.
   - ComfyUI's bundled templates are UI format; `python -m localgen_mcp.ui_to_api` converts them headless, and `scripts/make_trellis2_workflow.py` is a worked example.
3. Tools choose a workflow by modality and by which image slots the caller filled, so a new model is reachable through `model=` without touching the server. A new modality also needs a tool in `server.py`.

## Development

```bash
uv venv --python 3.12 /ai/venvs/localgen-mcp
uv pip install --python /ai/venvs/localgen-mcp/bin/python -r setup/mcp-requirements.lock -e .
/ai/venvs/localgen-mcp/bin/python -m pytest      # no GPU needed: the backend tests run against a fake ComfyUI
scripts/mcp_call.py --list                       # talk to the server over stdio like an MCP client would
PYTHONPATH=$PWD/src scripts/mcp_call.py --list   # the server gets this environment, so this runs a checkout's code
scripts/mcp_call.py generate_image '{"prompt": "a red apple"}'
blender -b -P scripts/render_glb_views.py -- mesh.glb out/mesh 512   # front, left, back, right stills of a GLB
blender -b -P scripts/render_glb_views.py -- lods.glb out/lod2 512 0 1 robot_LOD2   # one object only, wireframe on
```

The backend tests launch the real supervisor against `tests/fake_comfy/main.py`, a stand-in that answers the same endpoints and records its command line. They cover:
- the cold start, which must bind to localhost and run with `--offline`;
- reuse of a running backend;
- timeout extension and clamping;
- idle shutdown on a custom timeout;
- an extension from a second process being honoured;
- a busy queue keeping the backend up;
- four concurrent cold callers starting exactly one backend;
- `stop_backend` guarding a busy queue;
- an unmanaged ComfyUI being left alone.

## Credits

The server skeleton (FastMCP, stderr logging, signal handling) follows the layout of the author's other MCP servers. The ComfyUI client and the Z-Image Turbo graph are ported from a standalone Z-Image CLI written for the same install. Workflow graphs follow ComfyUI's own templates.

## License

MIT, see [LICENSE](LICENSE). Model weights are under their own licenses, listed above.
