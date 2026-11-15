# Model choices

Decided 2026-11-15 for an NVIDIA RTX 3080 Ti (12 GB VRAM), 62 GB RAM, ComfyUI 0.45.0 at commit `e08cc00304cd26879a23936886b947dd8a957966`.

Criteria, in order: fits 12 GB VRAM (quantized weights and ComfyUI's dynamic offload are acceptable), quality, a license that allows commercial use, native ComfyUI support (no custom nodes, so no extra code that might phone home), and downloadable without an account so a rebuild needs no tokens.

Practical constraint measured on 2026-11-15: the desktop session and other running applications already held 4.7 GB of VRAM, so the models must work with roughly 7 GB of free VRAM, not 12. RAM was 27 GB available. Anything that needs 40 GB or more of system RAM to offload is a poor fit for a shared machine.

## Summary

| Modality | Default model | Weights downloaded | License | Content filter |
|---|---|---|---|---|
| Image | Z-Image Turbo (already installed) | 0 GB new | Apache 2.0 | None in the weights |
| Video | Wan 2.2 TI2V 5B (fp16) + UMT5 XXL fp8 + Wan 2.2 VAE | 18.15 GB | Apache 2.0 | None in the weights |
| Mesh | TRELLIS.2 4B (int8 ConvRot) + DINOv3 L + BiRefNet + TRELLIS.2 shape and texture VAEs | 8.96 GB | MIT | None in the weights |
| Image edit (reference images) | FLUX.2 Klein 4B fp8 + FLUX.2 VAE, reusing the Qwen3 4B text encoder already on disk | 4.41 GB | Apache 2.0 | None described on the model card; not tested |
| Mesh from several views | Pixal3D multi-view (int8 ConvRot), reusing every TRELLIS.2 support file | 5.58 GB | MIT | None in the weights |

Total new download: 27.11 GB for the three defaults, plus 9.99 GB for the two reference-image models added later the same day (37.10 GB), into `/ai/ComfyUI/models/` (the standard ComfyUI folders, no symlinks). Sizes and SHA256 come from the Hugging Face API and are pinned in `setup/models.manifest`.

## Image: keep Z-Image Turbo

Z-Image Turbo (6B, Alibaba Tongyi) is installed and proven on this machine (about 70 s cold, about 8 s warm at 1024x1024, 8 steps). The ComfyUI templates bundled with this ComfyUI version also offer Flux.2, Qwen-Image 2512, Krea-2 Turbo, Ernie Image and others, but none is clearly better for 12 GB: Flux.2 Dev and Qwen-Image are 20B class models that need heavy offload, and Flux.1/Flux.2 Dev use a non-commercial license. Z-Image Turbo is Apache 2.0 and has no content filter in the weights, which is the reason it was chosen. No change.

## Video: Wan 2.2 TI2V 5B

Candidates checked (file sizes from the Hugging Face API, model lists from the ComfyUI 0.45 workflow templates):

| Candidate | Weights | License | Fit on this machine |
|---|---|---|---|
| **Wan 2.2 TI2V 5B** fp16 | 10.0 GB model + 6.74 GB UMT5 fp8 + 1.41 GB VAE | Apache 2.0 | Runs with offload of the text encoder; one model for both text to video and image to video |
| Wan 2.2 14B T2V/I2V fp8 + LightX2V 4 step LoRAs | 2 x 14.29 GB per task (high and low noise experts), 1.23 GB per LoRA, plus the same UMT5 | Apache 2.0 | Better motion and detail, but each expert is larger than the free VRAM, so every step streams weights from RAM; t2v and i2v need different expert pairs (57 GB for both). Second option for later |
| LTX-2.5 22B distilled int8 ConvRot | transformer plus a 15.4 GB Gemma 4 12B text encoder, about 70 to 80 GB in total | LTX-2.x Community License (revenue threshold for commercial use) | Gated repo (needs a Hugging Face login and license acceptance, so the rebuild would need a token). A 12 GB report shows 11.4 GB VRAM and a 45 GB system RAM peak, about 7 minutes per clip. Too heavy for a shared 62 GB machine |
| HunyuanVideo 1.5 720p | 16.65 GB fp16 model + 9.38 GB Qwen 2.5 VL 7B fp8 | Tencent Hunyuan Community License (excludes the EU, UK and South Korea; acceptable use policy) | Larger than Wan 5B for similar quality class; more restrictive license |
| MiniMax H3, Kandinsky 5 Lite | MiniMax H3 needs a 32B Qwen3-VL text encoder | various | Too large or less established |

Decision: **Wan 2.2 TI2V 5B**. It is the only candidate that does both text to video and image to video in one model of a size that fits this card, it is natively supported (`Wan22ImageToVideoLatent`), Apache 2.0, ungated, and its weights carry no content filter. Alibaba states it produces 720P at 24 fps and "5-second 720P video in under 9 minutes" on a single consumer GPU (RTX 4090). On a 3080 Ti with about 7 GB free it will be slower; measured numbers are in the README after the acceptance tests. The 14B pair with the LightX2V 4 step LoRAs is the documented upgrade path if quality is not enough; it is not in the default manifest.

Content restrictions: Wan 2.2 has no safety filter in the weights or the ComfyUI nodes. The Apache 2.0 license has no acceptable use policy; the model card asks users not to generate harmful or illegal content. Training data is filtered, so explicit content is weak without community LoRAs.

## Mesh: TRELLIS.2

| Candidate | Weights | License | Notes |
|---|---|---|---|
| **TRELLIS.2 4B** (Microsoft) int8 ConvRot | 5.25 GB model + 1.10 GB shape VAE + 0.95 GB texture VAE + 1.22 GB DINOv3 + 0.44 GB BiRefNet | MIT (code and weights) | Native in ComfyUI since v0.34 (shape stage, texture stage, PBR baking, remesh, UV unwrap, all in PyTorch, no nvdiffrast). Outputs a textured GLB |
| Pixal3D (Tencent ARC) int8 ConvRot | 5.58 GB + 0.66 GB MoGe, shares the TRELLIS.2 VAEs | MIT (code and weights, per the TencentARC and Comfy-Org model cards as of 2026-11-15; third-party components keep their own licenses, see the project NOTICE). An earlier read of the card showed a custom license; it no longer does | Native, same pipeline; Comfy says it "generally performs better than Trellis.2" for alignment to the input image. Optional second option, not in the default manifest because of the license ambiguity |
| Hunyuan3D 2.1 | 7.37 GB | Tencent Hunyuan Community License | Native ComfyUI nodes produce shape only (no texture node in core); texture painting needs a custom node with compiled CUDA extensions |
| TripoSplat | | | Outputs Gaussian splats, not meshes |

Decision: **TRELLIS.2**, using ComfyUI's native nodes and the official Comfy template graph (`3d_pixal3d_trellis2_image_to_model`) with the switch set to TRELLIS.2. MIT licensed, textured GLB output, no custom nodes. Microsoft states 24 GB is required for the reference implementation and gives H100 times of about 3 s at 512^3, 17 s at 1024^3 and 60 s at 1536^3. ComfyUI's int8 weights and offloading make it run in less; a community guide suggests 12 GB cards need reduced settings. The workflow exposes the upsample resolution so it can be lowered if VRAM runs out. Measured numbers on this card are in the README.

Content restrictions: none in the weights. Background removal (BiRefNet) runs first, so the input should show one object.

Text to mesh is provided by chaining `generate_image` (Z-Image Turbo, with a prompt suffix asking for a single object on a plain background) into the mesh workflow.

## Reference-image models (added 2026-11-15)

Reference-image input was requested after the first round. What the defaults already accept: Wan 2.2 TI2V 5B takes one start frame (`Wan22ImageToVideoLatent.start_image`); TRELLIS.2 takes one image of the object. Z-Image Turbo is text only. Two additions, both checked against the free VRAM (about 7.8 GB with the desktop and other apps loaded):

| Candidate | Weights | License | Decision |
|---|---|---|---|
| **FLUX.2 Klein 4B** fp8 (Black Forest Labs) | 4.07 GB model + 0.34 GB VAE; text encoder is `qwen_3_4b.safetensors`, already downloaded with Z-Image | Apache 2.0, ungated | **Added.** Text to image and editing with one or more reference images in one native model (ComfyUI template `image_flux2_klein_image_edit_4b_distilled`). The model card quotes 13 GB VRAM for bf16; the fp8 weights are half that |
| **Pixal3D multi-view** int8 | 5.58 GB | MIT | **Added.** Mesh from front, left, back and right views of one object (`Pixal3DMultiViewConditioning`, template `3d_pixal3d_multi_views`). Same pipeline and VAEs as TRELLIS.2 |
| FLUX.3 Klein (int8) | 5.83 GB + 4.86 GB Qwen3-VL 4B | "other", gated (Hugging Face login and acceptance) | Not added: a rebuild would need a token, and the license terms were not reviewed |
| Qwen-Image-Edit 2511 int8 + Lightning LoRA | 20.50 GB + 9.38 GB Qwen 2.5 VL 7B | Apache 2.0 | Not added: the model is more than twice the free VRAM, so every step streams weights from system RAM |
| Qwen-Image 2.1 Turbo edit int8 | 7.26 GB + 9.35 GB Qwen3-VL 8B (+ optional 9.47 GB prompt enhancer) | "other" | Not added: at the limit of free VRAM with a large text encoder, license not reviewed |
| Flux.1 Kontext Dev | about 12 GB fp8 | FLUX.1 non-commercial | Not added: non-commercial, and Klein 4B covers the same use |

Source for sizes, revisions and SHA256: the Hugging Face API, pinned in `setup/models.manifest`. Model cards: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B , https://huggingface.co/Comfy-Org/flux2-klein , https://huggingface.co/TencentARC/Pixal3D .

## Sources

- Wan 2.2 TI2V 5B model card: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- ComfyUI repackaged Wan 2.2 weights (Apache 2.0): https://huggingface.co/Comfy-Org/Wan-2.2 and https://huggingface.co/Comfy-Org/Wan-2.1
- Wan 2.2 requirements overview (community): https://wan27.org/blog/wan-2-2-requirements-guide
- LTX-2.5 ComfyUI tutorial: https://docs.comfy.org/tutorials/video/ltx/ltx-2-5 ; 12 GB report (11.4 GB VRAM, 45 GB RAM, 6 min 55 s): https://www.nemovideo.com/blog/ai-video-workflows/ltx-2-5-local ; license and gating: https://huggingface.co/Lightricks/LTX-2.5
- HunyuanVideo 1.5 repackaged (tencent-hunyuan-community license): https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged
- TRELLIS.2: https://github.com/microsoft/TRELLIS.2
- TRELLIS.2 and Pixal3D native in ComfyUI: https://blog.comfy.org/p/trellis2-and-pixal3d-are-now-native and https://docs.comfy.org/tutorials/3d/trellis2
- TRELLIS.2 VRAM guidance (community): https://book.st-hakky.com/en/data-science/trellis2-native-vs-legacy-vram-guide
- Pixal3D: https://github.com/TencentARC/Pixal3D and https://huggingface.co/TencentARC/Pixal3D
- Hunyuan3D 2.1 repackaged: https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged
- Model lists: the ComfyUI workflow templates shipped with this install (`comfyui_workflow_templates_json` 0.1.133): `video_wan2_2_5B_ti2v.json`, `video_wan2_2_14B_t2v.json`, `video_ltx2_5_t2v.json`, `video_hunyuan_video_1.5_720p_t2v.json`, `3d_pixal3d_trellis2_image_to_model.json`, `3d_hunyuan3d-v2.1.json`
