# localgen-mcp

A local MCP server that lets agents generate images, video and 3D meshes on this machine's GPU, fully offline, through one shared [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend that starts on demand and shuts itself down when idle.

- **Localhost and offline only.** The managed ComfyUI binds `127.0.0.1` and always runs with `--offline` (no partner or API nodes, a restrictive CSP for its frontend). Hub lookups are disabled with `HF_HUB_OFFLINE=1`. The network is used only by the setup scripts, to fetch software and weights.
- **Nothing resident when idle.** The MCP server imports no ML libraries. ComfyUI runs only while there is work, and its process is killed after the idle timeout (default 300 s), which releases all VRAM and RAM.
- **One backend for every agent.** Each agent session runs its own MCP server process; they all share a single ComfyUI instance, coordinated with file locks.
- **Returns file paths**, plus the seed and settings used, so results can be reproduced.

## Tools

| Tool | What it does |
|---|---|
| `generate_image` | Text to image (Z-Image Turbo). With `reference_images` (up to 4 paths), edits or combines them as the prompt says (FLUX.2 Klein 4B). With `input_image` and `denoise`, image to image; `mask_image` limits the change to a region and leaves the rest byte for byte (below). `count` queues several at once with one model load. |
| `generate_video` | Text to video, or image to video with `input_image` as the first frame (Wan 2.2 TI2V 5B, MP4). |
| `generate_mesh` | Textured GLB from `input_image` (TRELLIS.2), or from `prompt` alone (an image is generated first). With `left_image`, `back_image` and/or `right_image`, builds from several views (Pixal3D multi-view); `framing` picks how the views are read (below). |
| `retopologize_mesh` | Game weight copy of any textured GLB: quad remesh (Instant Meshes) or decimation to `target_faces`, new UVs, and base color, roughness, metallic and normal maps baked from the original. Writes a GLB and an FBX (the FBX keeps real quads). `lods` adds a LOD chain sharing one texture set. Headless Blender on the CPU, about 10 to 30 s. `generate_mesh` runs it too when given `retopo_faces` or `retopo_lods`. |
| `texture_mesh` | Texture an existing mesh (GLB, glTF, OBJ or STL) from a prompt and optional `style_image`: fresh UVs, a painted base color projected from several views, AO, a bump-derived normal map, roughness and metallic. Writes a GLB, an FBX and the painted views. About 35 s on the GPU plus a few seconds in Blender. |
| `generate_parts` | Build a model as separate part meshes from a front view concept and a box for each part, assembled in place and rigidly rigged (one bone per part, pivoting at the joints). Writes a GLB and an FBX. About 1 to 2 minutes per part. |
| `generate_material` | Tileable PBR surface material from a prompt (optional `reference_image`, or `tile_image` to make an existing picture tile): a seamless base color plus height, normal (OpenGL +Y, or DirectX), roughness and AO maps, a 3x3 tile preview and a lit Cycles render. 1024 or 2048 px; about 25 s warm at 1024 plus 5 to 15 s in Blender. |
| `generate_turnaround` | Consistent front, left, back and right views (optionally three-quarter views too) of one object from a concept image or a prompt: flat lit, plain background, framed at one scale for Pixal3D's rig, plus a sheet. By default each view is painted over a render of a TRELLIS.2 guide mesh from the rig's cameras, so flat objects turn properly. `mesh=True` meshes them through `generate_mesh`'s multi-view path. |
| `remove_background` | Cut the subject out of 1 to 64 images in one batch (BiRefNet): RGBA PNGs with a transparent background. Optional `trim` to the subject with `padding`, `square` canvas, output `size`, or a solid `background` colour. For UI icons, item sprites and compositing. |
| `speak` | Text to speech for game dialogue (Kokoro-82M, English, on the CPU): 24 kHz WAV plus OGG, in one of 28 voices or a blend, with an optional `effect` (`radio`, `robotic`, `glitch` for a voice struggling on low power). Needs `install.sh --with-tts`. |
| `list_voices` | The installed voices (accent, gender, quality grade), the effects, and whether ffmpeg was found. |
| `rig_creature` | Soft rig for an organic creature: bone chains (`chain` for tentacles, tails and vines; `spine`; `wings` for rays and other flat flyers; `auto` picks) with smooth skin weights, at most 4 per vertex, so engine scripts or physics can bend it. Writes a GLB and an FBX. Headless Blender on the CPU, a few seconds. |
| `upscale_image` | Upscale a PNG or JPEG 2x or 4x with a super resolution model: Real-ESRGAN x4plus for concept art and general images, or 4x-PBRify V4 (`model="pbrify-v4"`) for game textures. Keeps transparency. A few seconds to a minute. |
| `generate_sprite` | One pixel art game sprite on a transparent background at an exact size (default 64 x 64): drawn at 1024, cut out, pixelized to a palette with a 1 pixel outline. Writes the sprite, an 8x preview and the full resolution cut-out for hand fixes. `views="turnaround"` makes front, left, back and right at one scale. |
| `generate_animation` | An animation strip for one character (idle, walk, chop, attack, hurt, die, or your own poses): FLUX.2 Klein edits of the base drawing, one shared palette, per frame consistency metrics with retries, a strip PNG, the frames, a JSON and a GIF preview. With `pose_clip`, every frame copies the pose of a baked mocap clip (below). |
| `import_sprite_sheet` | Cut a sprite sheet drawn anywhere (another image model, an artist, another tool) into frames on one ground line and one torso column, key out the flat background (or BiRefNet), and pixelize them with one palette: the same strip, frames, JSON and GIF as `generate_animation`. Checks the frame count and grid (below). CPU only. |
| `render_pose_guides` | A baked pose clip's poses drawn as one sheet image (and one image per pose), sized to match another model's canvas, to send as a pose reference with a prompt hint. CPU only. |
| `bake_pose_library` | Bake FBX or BVH mocap clips into the pose library once (headless Blender): about 20 humanoid joints per frame, positions and local rotations, in place, with an action tag, loop flag, weapon hand and key frames (wind-up, impact, recover). |
| `list_pose_clips` | The clips in the pose library, with their action, length, impact frame and key frames. |
| `generate_tileset` | Terrain and building tiles on one grid and one palette, packed on a power of two sheet with an atlas. Tile kinds are yours to name; ground kinds can be seamless. |
| `pack_sheet` | Pack sprites or frames into a Unity friendly sheet (uniform cells, power of two) with a JSON atlas and a labelled contact sheet. CPU only. |
| `pixelize` | The pixel pass on its own: any image (a drawing, a render, a hand painted file) to pixel art at an exact size and palette. CPU only unless `remove_background`. |
| `status` | Whether the backend is running and managed, the effective idle timeout, queue length, GPU memory, and which models are installed. |
| `start_backend` | Warm the backend ahead of a batch, optionally with a longer `idle_timeout_seconds`. |
| `stop_backend` | Stop it now to free memory. Refuses while jobs are queued unless `force=true`. |

Every generation tool takes optional `output_dir`, `seed`, `model` and `idle_timeout_seconds`. Outputs default to `/ai/output/localgen/<modality>/<date>/`. Long jobs send MCP progress notifications.

### Image to image and masked editing

`generate_image(input_image=..., denoise=..., mask_image=...)` starts from an image instead of noise.
- **Image to image:** the input is VAE encoded and sampled with partial `denoise` (0.05 to 1.0, default 0.6 for Z-Image Turbo). Z-Image holds on to the input strongly: at 0.3 and 0.6 only fine detail changes, at 0.9 the layout stays and colours and materials change, and 1.0 ignores the input. The prompt describes the whole result.
- **Denoise with few steps:** KSampler runs `steps` steps of an `int(steps / denoise)` step schedule, so at Z-Image's 8 steps any denoise from 0.89 up would be a full schedule from pure noise (measured: 0.9 gave an unrelated image). The graph picks 8 to 16 steps so the applied value is within 0.01 of the request up to 0.9; above that it is at most 0.94 (16 of 17 steps), so below 1 the input always counts. Klein's custom sampler gets the same treatment.
- **Size:** any input size works. It is worked on at about 1 megapixel with both sides a multiple of 16 and resized back, so the output has the input's size; `width` and `height` are refused with `input_image`.
- **Mask:** a black and white image, white = change, black = keep, read from its red channel and stretched to the input if its size differs. Sampling is limited to the mask (`SetLatentNoiseMask`), then the result is pasted over the original (`ImageCompositeMasked`) with the mask grown and feathered by 8 pixels. Every pixel more than about 18 px outside the white area is the original's, byte for byte (measured below). To replace what is in the region use `denoise` 1.0; to recolour an object, Klein follows the prompt better than Z-Image, which at 0.9 keeps some of the old colour.
- **With `reference_images`** (or `model="flux2-klein-4b"`): FLUX.2 Klein edits `input_image` as image 1, with the references as images 2 to 4 (so at most 3). It starts from the input's latent too, with `denoise` default 1.0, which is Klein's usual reference guided edit; `mask_image` limits it the same way.
- Refused combinations: `mask_image` or `denoise` without `input_image`, and `width` or `height` with it.
- Measured through the MCP server on a 1024x1024 image (Z-Image at 0.9, and Klein) and a 1001x757 one with a 433x311 mask (Z-Image at 1.0): no pixel more than 18 px outside the mask changed; from 16 to 18 px at most by one level; from 8 to 16 px by at most 8 (34 for Klein). Without the composite step, 99% of the pixels outside the mask changed by up to 22 levels (the VAE round trip).
- The input's alpha channel is dropped (the output is RGB). Like every input image, the mask is uploaded only after a content check.

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

### Turnarounds

`generate_turnaround` makes the views a multi-view mesh needs from one concept image (or from a prompt, generating the concept first).
- **Clean front:** FLUX.2 Klein first redraws the concept as a clean front view. Every prompt asks for flat even lighting, no cast shadows and a plain background, and names which way the object faces: the left view shows the object's own left side, facing image-left, as in Pixal3D's rig. `subject` names the object and its parts; parts named there turn up in more views.
- **`guide="mesh"`, for flat, winged or wide objects:** the other views are painted over renders of a guide mesh, so their pose comes from 3D, not from the image model's idea of the object.
  1. TRELLIS.2 meshes the clean front (50k faces, 1024 px textures; `generate_mesh`'s path). If that fails (a furry front can make TRELLIS.2 extract tens of millions of faces, whose remesh then runs out of GPU memory), it retries from the concept and says so in `warnings`; `guide_source="concept"` starts from the concept. The concept must then show the object from the front.
  2. Blender renders the guide mesh from the rig's own cameras (`src/localgen_mcp/blender/render_rig_views.py`, rig in `_rig.py`, shared with `scripts/render_glb_views.py`): colour on `background`, a uniform white world for flat light. One mesh and the rig's cameras mean one scale and the rig's framing in every view. TRELLIS.2 does not always turn a mesh's front to the rig's front (a test robot came out facing sideways, so the "front" render showed its side), so the mesh is first rendered small from the four axis aligned yaws, each compared with the image it was made from (silhouette aspect, overlap and colour, after cropping both to the silhouette), and the cameras turn by the best yaw (`guide_yaw` in the result). Front and back cannot be told apart when TRELLIS.2 made the back a copy of the front, which then does not matter.
  3. Klein paints each view image to image from its own render (`guide_denoise`, default 0.6), references [render, clean front, concept], with a prompt to keep the exact shape, silhouette and pose and only improve the surface to match the concept. The front is painted from its render too, so it is framed like the others; the clean front stays a reference. The painted views are used as rendered, with a BiRefNet mask as alpha.
  - Why: drawn from an empty latent, Klein redraws flat or winged objects facing the camera in every view. A Shade Moth (a spiky spore body, two wide fluffy wings, tendrils) came back with the wings spread flat in the side views, and Pixal3D built a four winged cross from them. `texture_mesh` had the same drift and the same cure.
  - `guide_denoise`: on the moth, 0.5 changed little beyond the render, 0.6 kept every silhouette and added surface detail, 0.75 began growing a wing patch back into the side view.
  - The result adds `guide_mesh` (a usable single view TRELLIS.2 mesh on its own), `guide_renders`, `guide_source`, `clean_front` and `stage_seconds`.
- **`guide="none"` (default), for standing objects:** Klein draws left, back and right (and with `three_quarter=True` the four diagonals) from references [clean front, concept] from an empty latent, and the views are framed as below. It is faster (no TRELLIS.2 run or renders) and better on standing objects (below), but flat objects come out facing the camera.
- **Framing (`guide="none"` only):** each view gets a BiRefNet mask and is scaled so its silhouette has the same height as in every other view, then all views are cropped by one square box around the union of their silhouettes. So every view has one scale, the object is centred, and the widest view spans 1/1.1 of the frame, as the rig wants. This assumes the object's height is the same from every side, which holds for a standing object seen level. The views are RGBA PNGs (alpha is the mask), plus a sheet of them side by side.
- **Meshing:** `mesh=True` feeds the four main views to `generate_mesh` with `framing="as_framed"` and the views' own alpha. With `guide="mesh"` the result also carries the guide mesh.
- **Limits:** the views are drawn separately, so small asymmetric parts can drift between them. On the test robot, with a generic `subject`, its one antenna came out doubled in the back view, or on the wrong side, depending on the seed; large parts (head, cargo cage, arms, treads) stayed consistent. Naming the part and its side in `subject` ("exactly one antenna, behind the right side of its head (its own right)") gave one antenna, on the right side, in every view. Starting the back view from the mirrored front did not help (at denoise 0.7 the back kept the face). `symmetric=True` mirrors the left-hand views into the right-hand ones, which keeps left and right consistent for symmetric objects only. Look at the views before meshing.
- **Limits of `guide="mesh"`:** the views can only be as right as the guide mesh, and TRELLIS.2 guesses the hidden back from one front view (the moth's back came out as a copy of its front). Klein at 0.6 keeps the shape but matches the concept's surface only loosely: on the 2026-10-11 Shade Moth the wings lost most of the violet and green speckle, and on the 2026-10-30 one the glowing wing edges came out faint. It takes 3 to 6 minutes instead of about 1.5, with two model swaps (Klein, TRELLIS.2, Klein).
- **Tested (Shade Moth, 2026-11-15):** with `guide="none"` every view came back facing the camera with the wings spread, and Pixal3D built a four winged cross. With `guide="mesh"` the side views show the wings edge-on and the body in profile, and `mesh=True` gave a moth with two wings, thin from the side: bounding box 0.988 x 0.389 x 0.844 (TRELLIS.2 from the concept alone: 0.996 x 0.315 x 0.805). The later design (two wing pairs, two tail tendrils) gave 0.774 x 0.246 x 1.0.
- **Tested (robot, 2026-11-15), why `none` stays the default:** on a salvage robot with a camera head, `guide="mesh"` got the front and the side poses right (after the facing match), but its back view showed the camera face, because TRELLIS.2 had copied the front onto the guide mesh's back; its head was shallower in profile than the concept's, and the surface was softer. `guide="none"` on the same concept gave a real back (no face), a deeper head closer to the concept, and sharper detail. So the guide helps where the image model cannot turn the object (flat, winged, much wider than deep) and costs accuracy where it can.

### Background removal

`remove_background` takes a list of images and returns one PNG per image, the subject opaque and the background transparent.
- **No options:** each cut-out is the size of its input and lines up with it, ready for compositing.
- **`trim=True`:** crops to the subject's bounding box, plus `padding` on every side as a fraction of the subject's longer side (default 0.05; 0 for a tight crop). The result reports the box found, in input pixels.
- **`square=True`:** centres the result on a square transparent canvas. **`size`:** scales the longer side to that many pixels (with `square`, size x size). **`background="#rrggbb"`:** flattens onto a solid colour instead (RGB PNG).
- Typical calls: a UI icon is `trim=True, square=True, size=256`; an item sprite is `trim=True, padding=0`.
- **How:** one ComfyUI prompt holds a chain per image, so BiRefNet loads once for the batch: `RemoveBackground` gives the foreground mask, and `JoinImageWithAlpha` gets `InvertMask` of it, because it inverts its alpha input (it expects LoadImage's 1 - alpha convention). With any option set, a second prompt without a model crops, pads, scales and flattens each cut-out using native nodes (`ImageCrop`, `ImageCompositeMasked`, `ImageScale` with Lanczos). No native node returns a bounding box as numbers, so pass 1 also saves a 512 px copy of the mask and the server reads the box from it with the standard library, rounding outwards.
- **Limits:** the edge colour is the original pixel colour, so semi-transparent edges (hair, fur) keep a trace of the old background; a plain background that contrasts with the subject gives the cleanest result. Mask values under about 5% count as background when finding the box. Only the first frame of an animated GIF or WebP is used. BiRefNet's mask tops out just under 1, so the subject's alpha is 253 or 254 rather than 255. With layout options the job is two prompts, and on a busy shared backend each waits its turn in the queue.

### Speech

`speak` turns one line of English into a WAV file for game dialogue. It does not use ComfyUI or the GPU, so it runs beside image and video jobs.
- **Model:** Kokoro-82M v1.0 (Apache 2.0) on the CPU, in its own venv (`<root>/venvs/localgen-tts`). The MCP server runs `src/localgen_mcp/tts/runner.py` with that venv's Python and reads its final `TTS_RESULT {json}` line, the same pattern as the Blender scripts. The runner gets `HF_HUB_OFFLINE=1` and a CPU only PyTorch build.
- **Time:** about 5 s per line on 4 threads, most of it loading the model; each call starts a fresh process, so nothing stays in memory.
- **Voices:** 20 American and 8 British English voices; `list_voices` gives Kokoro's quality grade for each (`af_heart` A, `af_bella` A-, `bf_emma` and `af_nicole` B-, the male voices C+ at best). A comma separated list of up to 4 voices with one accent is averaged into a new voice.
- **Effects** (ffmpeg filter graphs in `tts/effects.py`; `intensity` 0 to 1 scales each):
  - `radio`: band limited to about 300 to 3200 Hz, compressed, light crushing and hiss.
  - `robotic`: a constant phase spectrum (a flat, buzzing monotone) with a short metallic echo.
  - `glitch`: for the low power lines. Stutters repeat the onset of some words (placed with the model's word timings), with dropouts, bursts of heavy bit crushing and light crushing throughout. `seed` reproduces a take.
  - The effected take is brought back to the dry take's peak level, and the dry take is kept as `<name>_dry.wav`.
  - Without ffmpeg, the runner renders approximations with numpy and writes WAV only.
- **Output:** 24 kHz mono 16 bit WAV and an OGG Vorbis copy, named by `name` (for example `pathfinder_hull_40_lowpower`). The result gives the duration, sample rate, RMS and peak level, and word timings (usable for subtitles).
- **Limits:** English only. Very short lines (a word or two) are weaker; Kokoro's model card suggests bundling them. No voice cloning; see [docs/model-choices.md](docs/model-choices.md) for why Chatterbox was not chosen. Levels and durations are checked automatically; only a listener can judge the voice itself.

### Soft rigs for creatures

`rig_creature` rigs organic meshes for deformation driven in the engine (shaders, Verlet chains, springs); it makes no animation clips. Run it on a `retopologize_mesh` result: raw generated meshes are heavy and full of stray shells.
- **Presets.**
  - `chain`: `bones` (default 8) along the main axis or a given `axis`, rooted at `root_end` (default: the lower end of a vertical chain, else the thicker end). Joints are slice centroids, so the chain follows a gently curved mesh.
  - `spine`: the same along a body.
  - `wings`: the thin axis is the surface normal, and the left-right axis is the one the mesh is mirror symmetric across. A spine runs down the middle, and `wing_chains` chains of `wing_bones` bones fan out on each side to the wing edge.
  - `auto`: picks from the proportions and reports why.
- **Names.** `tentacle_01` (or `<name>_01`), `spine_01`, `wing_L1_01` and the mirrored `wing_R1_01`, numbered from the root and, for wing chains, from the head. Every bone's local Y runs along it; rotating about local X bends it (for wings, towards the surface normal: a flap).
- **Weights.** Blender's heat weights first. They often fail on generated meshes: on the test manta, a spine rig left 100% of the vertices unweighted ("failed to find solution"). When more than 1% of the vertices are unweighted, or weighted mainly to a bone more than twice as far as their nearest, all weights are replaced by a smooth inverse distance falloff from the bone segments; a smaller share is patched. Vertices at the same position (split at UV seams) get the same weights, then each vertex keeps its 4 largest, normalised. The report gives the method, the heat failure shares, coverage and influences per vertex.
- **Output.** A GLB and an FBX in the rest pose, and a report of every bone (head, tail, parent).
- `scripts/render_glb_views.py ... --pose 'wing_.*=12'` renders a rig posed, framed like the rest pose, for before and after stills.
- **Limits.**
  - Joints come from straight slices, so a tentacle curled through more than about 90 degrees gets a chain that cuts the curve. Give it a straighter mesh or split it.
  - Distance weights are Euclidean, so parts that touch without being connected (a tentacle curled against the body) pull on each other.
  - On `wings`, the body is a fixed strip (12% of the half span each side of the middle). Anything outside it, such as a manta's head lobes, moves with the nearest wing chain.

### Upscaling

`upscale_image` runs a 4x super resolution model with ComfyUI's native `UpscaleModelLoader` and `ImageUpscaleWithModel` nodes.
- **Models:** `realesrgan-x4plus` (default) is a general upscaler that also cleans up mild noise and JPEG artefacts; it suits concept art, renders and photos. `pbrify-v4` is trained on game textures and keeps more fine surface detail, at about 3 times the run time. Both allow commercial use (BSD-3-Clause and CC0); see [docs/model-choices.md](docs/model-choices.md#upscaling-added-2026-11-15).
- **Scale:** 4 is the models' native scale. 2 downscales the 4x result by half with Lanczos, so one model serves both scales.
- **Alpha:** a PNG with transparency keeps it (`keep_alpha=false` drops it). The model upscales the colour, and the alpha channel is resized bilinearly. Colour hidden under fully transparent pixels is upscaled too, so edges look as clean as the source's.
- **Large images:** the upscale node works in 512 px tiles (halved if the GPU runs out of memory), so VRAM is not the limit. The result is limited to 8192 px per side: 2048 px inputs at 4x, 4096 px at 2x. Output is an 8-bit PNG named `<input>_x<scale>.png`.

### Pixel art sprites

Seven tools make 2D game sprites (two of them, `import_sprite_sheet` and `render_pose_guides`, for sheets drawn elsewhere: see [Sheets from another image model](#sheets-from-another-image-model)). They reuse the existing pipeline: the image workflows draw (Z-Image Turbo; FLUX.2 Klein for style references and animation frames), `remove_background`'s BiRefNet pass cuts out, the seamless material graph makes ground tiles, and `generate_turnaround` makes views. Only the pixel pass is new.

- **The pixel pass** (`pixelize`, used by all of them) runs in `src/localgen_mcp/pixel_ops.py`, a numpy and Pillow helper the server starts as a child process with its own interpreter (`python -I`), reading its final `PIXEL_OPS_RESULT {json}` line, the same pattern as the Blender scripts; the server process still imports no imaging library. Per image:
  1. **Place:** one scale (source pixels per sprite pixel) maps the subject onto the size x size cell. `fit="trim"` fits the subject's box, `"frame"` the whole image (one scale for images framed alike, as turnaround views are), `"fill"` stretches the whole image (opaque tiles). `anchor="bottom"` stands the subject on the bottom edge, centred on its mass. A fixed scale can be passed, which is how animation frames keep the base sprite's size.
  2. **Downsample:** a Gaussian blur (sigma = `blur` x scale / 2, `blur` 0.5 by default, premultiplied alpha) evaluated only at the cell pixel centres, which is a blur followed by nearest neighbour sampling. On the test sprites 0.5 kept eyes and line art; 1 and above blurred them away, and 0 aliases. With `wrap=True` the blur wraps, so a seamless image stays seamless. Alpha is cut to fully on or off.
  3. **Colour:** optional `saturation` change, then each pixel goes to the nearest `palette` colour in OKLab. A lone pixel whose colour none of its neighbours share, and which is close in colour to what surrounds it, takes that colour (a bright eye or highlight stays). Specks and pinholes in the silhouette are cleaned.
  4. **Outline:** a 1 pixel ring round the silhouette in a palette colour. `"auto"` (default) picks, for each outline pixel, a palette colour clearly darker than the darkest colour it borders, so a beard gets a dark brown edge and armour a dark grey one; `"darkest"` or `"#rrggbb"` use one colour. One palette slot is kept for it, so a sprite never uses more than `max_colors` colours.
- **Palettes:** `None` or `"auto"` extracts at most `max_colors` colours (default 32) from the image (k-means in OKLab; with `shading_steps`, hue families of that many tones each). A shipped palette by name: `"moriveil"` (`src/localgen_mcp/palettes/moriveil.gpl`, 32 warm colours: outline darks, skin, beard, a green skin ramp, wood, stone, grass, water, cloth and gold); more `.gpl` files dropped in that folder ship as more names. Or a list of `"#rrggbb"`, or a palette file: `.gpl`, `.hex` or `.txt` (one `#rrggbb` per line) or a `.png` whose colours are the palette. Palette files are read only from under the output folder (`LOCALGEN_OUTPUT_DIR`) or `<root>/tmp`; copy one there first. Text palettes must parse completely as colours (errors give a line number, never the line), and a helper failure returns a short message while its output goes to the server log, so the parameter cannot be used to read other files. With more entries than `max_colors`, the most used are kept.
- **Styles:** `style` is a preset or free text. `"cartoon"` (default): warm cartoon game art, chunky dark outlines, three tone cel shading, saturated. `"grim"`: muted dark roguelike, black outlines, two tones. `"painterly"`: soft brushwork, no outline. `"flat"`: flat colours, no shading, no outline. Free text ("a 1-bit monochrome dungeon crawler") becomes the prompt wording with neutral defaults. `outline`, `outline_color`, `shading_steps` and `saturation` override any preset, so a 32 pixel grim sprite or a 128 pixel painterly tile needs no code change: `generate_sprite(prompt="a hooded plague doctor", size=32, style="grim")`, `generate_tileset(kinds=[{"name": "moss_stone", "seamless": true}], size=128, style="painterly")`.
- **`generate_sprite`:** the prompt plus the style's wording, drawn at 1024 on a plain grey background (Z-Image Turbo; with `style_images`, up to 4 pictures whose art style to copy, FLUX.2 Klein), cut out, pixelized. Writes `<name>_<seed>.png`, `_x8.png` (preview), `_source.png` (the full resolution cut-out, to fix in Aseprite or any editor and run through `pixelize` again), `_raw.png` (the drawing) and `_palette.gpl` with an auto palette. `views="turnaround"` runs `generate_turnaround` on the drawing and pixelizes the four views with `fit="frame"` and one joint palette.
- **`generate_animation`:** the base is `sprite_or_prompt` (an image, best a `generate_sprite` `_raw.png`; a finished small sprite is scaled up with nearest neighbour to serve as the reference), or a prompt for `generate_sprite`. Every frame is a Klein edit of the base drawing (image 1, so the identity always comes from the base) with the previous frame as image 2 and a pose prompt for that frame. Presets give per frame poses, a frame rate and thresholds for `idle`, `walk`, `chop`, `attack`, `hurt` and `die`; `frame_prompts` gives your own poses. Frames are cut out in one batch and pixelized at the base sprite's scale with the base sprite's palette, so colours cannot drift. Each frame is measured against the base sprite: `hist_distance` (total variation distance of the colour histograms: 0 the same colours in the same shares, 1 nothing shared) and `iou` (silhouette overlap); and against the previous frame: `motion` (1 - silhouette overlap). A frame under the action's thresholds (`min_iou`, `max_hist_distance`, and `min_motion` so a frame that only repeats the one before it fails; walk 0.05, chop and attack 0.08, idle 0) is drawn again with a new seed and without the previous frame as image 2, since that reference pins the pose, up to `retries` (default 2) times, keeping the best attempt (passing first, then closest to the base). `consistency` in the result sums it up, including `static_frames`, frames whose silhouette changed by under 2%. Output: `<name>_<action>_<seed>_strip.png` (frames left to right), `_fNN.png`, `.json` (frame count, size, fps, loop, palette, thresholds, per frame metrics), `_preview.gif`, `_contact.png`, and each frame's drawing and cut-out.
- **Pose driven animation (`pose_clip`):** Blender runs once per mocap clip, in `bake_pose_library`; every later sprite sheet draws its pose guides in plain Python (`src/localgen_mcp/pose_math.py` and the pixel helper), so no Blender is needed to make an animation.
  1. **Bake:** `blender/bake_pose_clips.py` imports each FBX or BVH, maps its armature to the canonical joints (pelvis, spine, chest, neck, head, crown and face points, shoulders, elbows, wrists, hands, hips, knees, ankles, toes) by bone name rules covering Mixamo (with or without `mixamorig:`), Unity Humanoid, the Unreal mannequin (`upperarm_l`, `calf_l`), `B-` rigs with `.L`/`.R` suffixes, 3ds Max Biped and CMU BVH names (twist, roll, finger, IK and prop bones are skipped), and samples every frame. It stores each joint's position and its local rotation (the segment's rotation relative to its parent segment, as a change from rest; the rest offsets and these rotations give the positions back by forward kinematics) with the rest bone lengths, fps and source file. The figure is stood upright (a file read with the wrong up axis), turned to face the front camera, put on the floor, converted from centimetres when it is that tall, and kept in place (the pelvis pinned over the origin; `root_motion="keep"` keeps the travel). A rig that animates a bone above the pelvis with a turn and a scale (some packs key a root at 90 degrees and x100) is handled by reading the rest pose through that bone's first frame transform.
  2. **Metadata:** the action tag comes from `action` or the file name (walk, run, idle, attack, chop, hammer, hurt, die, block, jump); loop from `loop`, a "loop" in the name or a looping action; the weapon hand from `weapon_hand`, an `_L`/`_R` in the name, or the hand that moves most. Key frames from the motion: strikes get wind-up (the hand pulled back furthest against the strike), impact (the first frame the weapon hand all but stops after its fastest motion, or its most extended frame near the fastest when it never stops) and recover; hurt gets the recoil; die the frame the pelvis gets down; walks the two foot contacts and the passing frames.
  3. **Library:** one `<clip>.json` per clip plus `index.json` in `LOCALGEN_POSE_DIR` (default `/ai/data/localgen/poses`), outside the repository; baked clips and mocap files are never committed. `list_pose_clips` reads the index; `pose_clip` takes a clip name, an action tag (the first clip with it) or a clip file under the library, output or tmp folder.
  4. **Guides:** the clip is retargeted to `body` by scaling the rest offsets per segment group and keeping the rotations (`human`; `stocky`: short legs, broad, big head and torso; `small`: short, big head, long arms; `tall`; or a dict of factors `legs`, `feet`, `arms`, `torso`, `neck`, `head`, `shoulders`, `hips`, `thickness`), the pelvis height scaling with the legs and the feet put back on the floor. `exaggerate` scales every joint's rotation away from the clip's average pose (not the bind pose, which is often a T pose). An orthographic camera at `view` (front, back, left, right and the four 3/4 views, or an azimuth; "left" sees the character's left side) and `elevation` (0 side on, 90 straight down; 30 to 60 for a top down strategy view) projects it, and the figure is drawn at the size of the base drawing, standing where the base character stands (its first frame as tall as the cut-out's box, feet on its bottom edge; if a later frame would leave the canvas, one smaller scale is used for all). `guide_style="capsule"` draws a shaded mannequin far to near (near limbs over far ones, darker with distance, left limbs blue and right limbs orange, a dot for the face); `"skeleton"` an OpenPose coloured stick figure on black. `weapon` (`dagger`, `sword`, `axe`, `warhammer`, `spear`, `shield`, or a weapon and a shield) adds a plain prop in the weapon hand, its direction carried by the hand's rotation; a shield goes on the other hand.
  5. **Frames:** `frame_selection="auto"` takes frames evenly over one cycle of a loop (leaving out a last frame that repeats the first, and using the first cycle when the clip holds several, as a walk recorded over two strides) and the key poses of a one shot action, the impact always included (with more frames than keys the largest gaps are split; with fewer, impact, wind-up, start and recover win). Each frame is a Klein edit with the base drawing as image 1 and its guide as image 2 ("the character from image 1 in the exact pose of image 2"), never the previous frame; retries keep the guide. The palette pass and the metrics are the same as without a clip. The JSON adds `impact_frame` (index in the strip), `durations_ms` from the clip's timing (the GIF uses them; `fps` is their average unless given), and `pose` (clip, body, camera, clip frames, key labels, guide paths and a guide contact sheet).
  6. **Example:** `bake_pose_library(inputs=["/path/to/mocap/folder"])`, then `generate_animation(sprite_or_prompt="hero_raw.png", action="attack", pose_clip="attack", body="stocky", view="right", elevation=20, weapon=["sword", "shield"], frames=6)`.
- **`generate_tileset`:** `kinds` are yours: names (`"lava_pool"`) or `{"name", "prompt", "seamless", "transparent"}`. Seamless kinds come from the `generate_material` seam repaint graph (base colour only) and are pixelized with a wrapping blur; transparent kinds (the default) are objects cut out onto a transparent cell with the style's outline; `transparent: false` makes an opaque tile that fills the cell. `prompt` is a theme added to every tile. Without `kinds`, preset `"strategy"` gives grass, forest, stump, rock, gravel, water edge, wood and stone walls, gate open and closed, tower, home, hearth, farm empty and ripe, and raider camp. With an auto palette, one palette is extracted from all tiles together.
- **`pack_sheet`:** uniform cells (default the largest image), left to right and top to bottom, the sheet padded to power of two sides; smaller images sit bottom centre (`align`). The JSON atlas gives every cell's name, x, y, w, h from the top left and `unity_y` from the bottom left, plus Unity import settings: Sprite Mode Multiple, Grid By Cell Size with the cell as pixel size, Point filter, no compression. The contact sheet enlarges every cell on a checkerboard with its name, for review in Slack.
- **Examples:**
  ```
  generate_sprite(prompt="a stout dwarf worker with a braided ginger beard and a wood axe", palette="moriveil")
  generate_animation(sprite_or_prompt="/ai/output/localgen/sprites/dwarf_worker_101_raw.png", action="walk",
                     subject="a stout dwarf worker with a braided ginger beard", palette="moriveil")
  generate_tileset(prompt="a cosy medieval frontier", palette="moriveil",
                   kinds=["raider_camp", {"name": "farm", "prompt": "ripe wheat in rows", "transparent": false},
                          {"name": "grass", "seamless": true}])
  pixelize(image="painted_tree.png", size=32, palette="moriveil", remove_background=true)
  pack_sheet(images=["a.png", "b.png", {"name": "hero_idle_0", "path": "c.png"}], cell=64, columns=8)
  ```
- **Limits (read before relying on the output):**
  - Diffusion does not keep a character the same across frames, and the motion is the weak part. In the sample walk, Klein kept the dwarf's identity very well (colours, beard, axe, apron), but held one mid stride pose: with the previous frame as image 2, all six frames came out as nearly the same stride (silhouette change of 1 to 3% between frames), which is why the motion check and the no image 2 retry exist. Even without image 2 the legs change only a little. With `pose_clip` (a mocap walk, `body="stocky"`, side view guides) the same base drawing's legs followed the guides, alternating stride and passing poses from frame to frame, but Klein kept the base drawing's three-quarter view rather than turning the character to the guide's side view, and held the arm carrying the axe on the shoulder whatever the guide's arm did. A base drawn from the camera the guides use is the next thing to try. The metrics catch frames that are clearly off (a different silhouette, colours, or no motion), not small identity changes or wrong poses. Expect to redo the motion by hand.
  - All outputs are reference and placeholder quality until an artist passes over them. The pass gets the grid, palette and outline right, but not the pixel level decisions (clusters, hand placed anti aliasing, readable 1 pixel details) a pixel artist makes. Line art thinner than one sprite pixel at the target size is lost.
  - Seamless tiles wrap because the seam repaint makes the drawing wrap, but the material graph's texture wording pulls them towards realistic surfaces; the style wording is added, yet cartoon ground comes out less stylised than objects.
  - Not made: auto tiling transition sets (the 47 tile blob for grass to forest or grass to water). Every tile is one cell; buildings larger than a tile are drawn into one cell.

### Sheets from another image model

The local models are weak at animation (above). A hosted image model that draws a whole sheet in one picture, such as OpenAI's image models, keeps a character and a gait together far better, so the flow for animations can be: an agent asks such a model for a sheet, then hands the file to this server, which does the rest locally. This server never calls an outside service; the agent makes that call with its own credentials, and the sheet is just an image file here.

1. **Optional pose reference:** `render_pose_guides(pose_clip="walk", frames=6, body="stocky", sheet_size=[1536, 1024])` draws a baked clip's poses as one sheet (rows of up to 6, read left to right then top to bottom; `layout` sets the columns), each pose drawn at its cell's shape, side on and facing right by default (`view`). Its `prompt_hint` is wording to add to the prompt; the sheet goes to the other model as a second reference image after the character.
2. **Draw:** the agent sends the character (a `generate_sprite` `_source.png` or `_raw.png`, or any drawing) as the reference image, with a prompt like the template below.
3. **Import:** `import_sprite_sheet(image="walk_sheet.png", frames=6, rows=1, palette="moriveil", action="walk", gif_loops=3)` cuts it into the same outputs as `generate_animation`: strip, frames, JSON, atlas, contact sheet and GIF.

**What `import_sprite_sheet` does.** The background colour is read from the sheet's edge (a transparent sheet uses its alpha). Frames are found as row bands, then column runs, of the pixels that differ from it, read left to right and then top to bottom, so uneven spacing and a short last row are fine. Specks are ignored, a narrow detached piece (the far end of a tool) joins the nearer frame, and frames whose outlines touch are split at the narrowest column. `frames`, `rows` and `columns` say what to expect; when the count found differs, `on_count_mismatch="fail"` (default) stops and says how many frames it found in each row (the `_overlay.png` shows the boxes), and `"keep"` carries on with a warning. Each frame is then moved onto one square canvas: its lowest pixel on a common ground line, and the median column of its torso band (35 to 60% of its own height, which a swinging tool or a stride moves far less than its box or its centre of mass) on a common vertical. A flat grey ground shadow drawn under the feet is cleared first (`ground_shadow`). The background is keyed out (`key_threshold`, summed RGB difference, default 45), or with `background="birefnet"` each aligned cell is cut out by BiRefNet on the GPU, for a sheet whose background is not flat. Every canvas is pixelized with `fit="frame"`, so all frames share one scale and the motion between them survives, with one palette (the one given, or an auto palette from all frames together). The JSON records what was found (frames per row, background, canvas, ground line, anchor column, notes such as joined pieces) and each frame's box on the sheet and its motion from the frame before. `action` only sets the default fps and loop; `gif_loops` (also on `generate_animation`) is the number of cycles in the preview GIF, 3 being easier to review than 1.

**Prompt template** for a walk cycle sheet (replace the parts in brackets; for 12 frames use "laid out as two rows of 6 frames (frames 1 to 6 on the top row, 7 to 12 on the bottom row, read left to right)" and spell out the in between phases):

```
Pixel art game sprite sheet of the [character] in the reference image: a 6 frame walk cycle,
side view facing right, all 6 frames in one horizontal row, evenly spaced, same size and same
ground line in every frame. Classic walk cycle: contact, down, passing, up, then the same with
the other leg forward; arms swing opposite to the legs. Keep its exact design: [the features that
must not change: hair, clothing, colours, equipment]. [Art style: for example chunky warm cartoon
pixel art, dark 1 pixel outline, flat shading with one highlight and one shadow], readable at
[64] pixels. Plain flat light grey background, no text, no labels, no numbers, no shadows.
```

With a pose reference, add the `prompt_hint` and say that image 2 gives the poses ("the character from image 1 in the poses of image 2, one per frame"). A landscape canvas (1536 x 1024) suits one row of 6 and two rows of 6.

**Limits seen on real sheets** (a 6 frame walk in one row and a 12 frame walk in two rows of 6, from one hosted model, with the character's cut-out as the reference):
- The frames are not on an exact grid until imported: spacing and the baseline vary a little from frame to frame, and one sheet drew a grey ground shadow under every frame although the prompt asked for none. Both imported cleanly: 6 and 12 frames found, ground shadow cleared, torso and ground aligned.
- The model may ignore prop placement: asked for the axe hanging from the belt, it kept the axe on the shoulder as in the reference.
- The 12 frame sheet drifted towards a three-quarter view and smaller per frame motion (mean silhouette change 0.11 between frames against 0.16 for the 6 frame sheet, both at 64 px); the 6 frame sheet stayed side on.
- The reference image goes to whichever service the agent uses; this server sends nothing anywhere.

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
| Speech (optional, CPU) | Kokoro-82M v1.0, 28 English voices | Apache 2.0 | 0.34 GB, plus a 1.3 GB venv |
| Upscaling, general | Real-ESRGAN x4plus | BSD-3-Clause | 67 MB |
| Upscaling, game textures | 4x-PBRify UpscalerV4 (DAT2) | CC0-1.0 | 140 MB |

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
| Turnaround, from a concept image | Klein 4B, 4 views at 1024 px, framing and sheet | 81 s execution (129 s with the model swap) | 11.7 GB |
| Turnaround with three-quarter views, `symmetric` | 8 views, 5 of them sampled | 101 s execution | |
| Turnaround, `guide="mesh"` | clean front, TRELLIS.2 guide (50k faces), 3 or 4 Blender renders at 1024 px (4 threads), views painted | front 8 to 56 s, guide 38 to 166 s, renders 19 to 41 s, paint 89 to 114 s: 3 to 6 min (moths 3 min, robot 6 min) | |
| Retopology | 300k faces to 10k quads, 2048 px textures | 10 to 13 s (CPU) | none |
| Soft rig (`rig_creature`) | 8k quad manta, `wings`, 30 bones | 2 s (CPU) | none |
| Texture a mesh, warm | 4 views at 1024 px, 2048 px textures, 10k quad mesh | 34 s, plus 4 s in Blender | |
| Speech, one 4 to 6 s line (measured with the load average near 10) | Kokoro-82M, 4 CPU threads, any effect | about 5 s in the runner (mostly model load); 6 to 10 s per call over MCP | none (CPU) |
| Upscale 4x, `realesrgan-x4plus` | 1024x1024 to 4096x4096; 1536x1024 to 6144x4096 | 8.7 s; 12.1 s | |
| Upscale, `pbrify-v4` | 1024x1024, scale 2 (4x run, then Lanczos) | 25.0 s | |
| Idle shutdown | default 300 s | ComfyUI gone at 301 s idle; VRAM back to 3.6 GB, RAM freed (about 18 GB) | |

Times include the model swap when the previous job used a different model.

Mesh quality was checked by rendering each GLB from the four rig views in Blender (`scripts/render_glb_views.py`). The test was Pixal3D fed four renders of a TRELLIS.2 teapot. With `framing="as_framed"` it reproduced the original's proportions: a bounding box of 0.993 x 0.609 x 0.654 against 1.0 x 0.614 x 0.654, with one handle and one spout. With `crop` the same views produced a body as wide from the side as from the front.

## Install or rebuild

Requirements: Linux, an NVIDIA GPU and driver, `git` and [`uv`](https://docs.astral.sh/uv/). uv provides Python 3.12. Blender is optional: only retopology needs it, and `install.sh` reports whether it was found. ffmpeg is optional too: `speak` uses it for effects and OGG output.

```bash
setup/install.sh --root /ai            # ComfyUI, both venvs, directories, all model weights
setup/install.sh --root /ai --dry-run  # show what would be done, change nothing
setup/install.sh --skip-models         # software only
setup/download-models.sh --only video  # weights for one group (image, video, mesh, upscale); --verify rechecks SHA256
setup/install.sh --root /ai --with-tts # also the text to speech venv (CPU) and the Kokoro weights
```

What gets pinned:
- ComfyUI at commit `e08cc003` (0.45.0).
- Both venvs from frozen lock files (`setup/comfyui-requirements.lock`, `setup/mcp-requirements.lock`).
- Every weight file by Hugging Face revision and SHA256 (`setup/models.manifest`). The upscale models exist only as GitHub release assets, so those entries name the release tag and are pinned by SHA256.
- With `--with-tts`: the text to speech venv from `setup/tts-requirements.lock` (every package hashed, CPU only PyTorch) and the Kokoro files in `setup/tts.manifest`, into `<root>/models/tts`.

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
| `LOCALGEN_TIMEOUT_PIXEL` | 600 s | Pixel pass helper (sprite tools) |
| `LOCALGEN_POSE_DIR` | `<root>/data/localgen/poses` | Baked pose clip library (`bake_pose_library`), outside the repository |
| `LOCALGEN_TTS_PYTHON` | `<root>/venvs/localgen-tts/bin/python` | Text to speech venv |
| `LOCALGEN_TTS_MODEL_DIR` | `<root>/models/tts/kokoro-82m` | |
| `LOCALGEN_TTS_THREADS` | 4 | CPU threads for synthesis |
| `LOCALGEN_TIMEOUT_TTS` | 600 s | |
| `LOCALGEN_FFMPEG` | `ffmpeg` on `PATH` | Effects and OGG output |

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
LOCALGEN_GPU_SMOKE=1 /ai/venvs/localgen-mcp/bin/python -m pytest tests/test_sprites.py -k gpu   # real backend: one sprite and a 4 frame idle
LOCALGEN_GPU_SMOKE=1 /ai/venvs/localgen-mcp/bin/python -m pytest tests/test_poses.py -k gpu    # a pose driven strike from a synthetic clip
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
