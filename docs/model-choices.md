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

## Sound effects and ambience: not added (checked 2026-11-15)

Requested for game audio: creaking metal, electrical hums, the clicking of an organism, cave drones, footsteps on metal. Nothing was built. The one model that fits (Stable Audio 3 Small SFX) is gated behind a terms acceptance and its license needs a registration step only the owner can do, so this is a findings section only.

What ComfyUI 0.45 (`e08cc003`) runs natively, from `comfy/supported_models.py`, `comfy_extras/nodes_audio.py`, `nodes_ace.py` and the bundled templates (`comfyui_workflow_templates_json` 0.1.133, `audio_*.json`):
- **Stable Audio Open 1.0** (`StableAudio`, T5 base encoder) and **Stable Audio 3** (`StableAudio3`, T5Gemma encoder), with `EmptyLatentAudio`, `KSampler`, `VAEDecodeAudio` and `SaveAudio`/`SaveAudioAdvanced` (FLAC, MP3, Opus). The SA3 template samples 8 steps, `lcm`, cfg 1, and can expand the prompt with Qwen 3.5 2B using an "SFX" preset.
- **ACE-Step 1 and 1.5**, **YuE2** and **MiniMax Music 3**: music models (songs, instrumentals).
- **MMAudio**: only its VAE is in core (used by LTX audio); there is no native MMAudio sampler.
- Everything else sound related in the templates (`api_elevenlabs_text_to_sound_effects`, `api_mirelo_text_to_sfx`, `api_pika_sfx`, ...) is a paid partner API node, which the offline backend does not load.

| Candidate | Weights | License (code / weights) | Gated on Hugging Face | Verdict |
|---|---|---|---|---|
| **Stable Audio 3 Small SFX** (Stability AI, 2026) | 2.27 GB checkpoint + 1.19 GB T5Gemma encoder; up to 2 minutes, effects and short ambience | Stability AI Community License; the T5Gemma encoder is under the Gemma Terms of Use | `stabilityai/stable-audio-3-small-sfx` is gated ("auto": name, email, country, organisation, intended use, accept the license and the Gemma terms). `Comfy-Org/stable-audio-3` rehosts the same files ungated | **The right model, not added**: downloading from the ungated rehost would sidestep the gate the owner set, and commercial use needs a registration (below) |
| Stable Audio 3 Medium (and Base) | 9.22 GB checkpoint + the same encoder | same | same | Music first, larger; Small SFX suits this use better |
| Stable Audio Open 1.0 (2024) | 4.85 GB + 0.85 GB T5 base (Apache 2.0) | Stability AI Community License | upstream gated; `Comfy-Org/stable-audio-open-1.0` rehost ungated | Same license and gate problem, older and weaker than SA3 |
| ACE-Step 1.5 | 4.79 GB turbo model + 1.19 GB Qwen 0.6B + 0.34 GB VAE (Comfy repack) | MIT upstream, Apache 2.0 repack | no | Commercially fine and native, but a music model: no footsteps, clicks or creaks |
| MiniMax Music 3 | int8 model + encoder | MiniMax-Music3 Community License (MIT style, plus conditions not reviewed) | no | Music only |
| YuE2 | 3B int8 | CC BY-NC 4.0 | no | Non-commercial, music |
| MMAudio | | MIT code, CC BY-NC 4.0 weights | no | Non-commercial; no native sampler |
| AudioLDM 2 | | CC BY-NC-SA 4.0 weights | no | Non-commercial; not native |
| Tango 2, TangoFlux | | Tango 2: CC BY-NC-SA 4.0. TangoFlux: no license tag on the repo, and it uses the Stable Audio Open VAE, so at least that license applies (not reviewed further) | no | Non-commercial or unclear; not native |
| AudioX | | CC BY-NC 4.0 weights | no | Non-commercial; not native |
| Meta AudioGen | | CC BY-NC 4.0 weights | no | Non-commercial; not native |
| Sony Woosh (2026) | | Apache 2.0 code, CC BY-NC weights (Sony's announcement) | not checked | Non-commercial; community node only |
| HunyuanVideo-Foley | | Tencent Hunyuan Community License | no | Video to audio, territorial exclusions; not native |
| ThinkSound | | Apache 2.0 | no | Mainly video to audio; custom nodes only |
| EzAudio | | MIT (weights card) | no | The only ungated, permissively licensed text to sound effect model found, but not native: it would need its own inference code or a custom node, outside this project's no custom nodes rule. Quality on this kind of prompt was not tested |

### Stability AI Community License, as it applies here

From https://stability.ai/community-license-agreement (read 2026-11-15; not legal advice):
- Commercial use is allowed while the licensee and its affiliates (over 50 percent control), together, make less than USD 1,000,000 in annual revenue, counted whether or not the revenue comes from the model. Above that the license ends and an enterprise license must be requested (granted at Stability's discretion); the weights and derivatives must then be deleted.
- **"You must register with Stability AI"** for commercial use.
- Products or documentation must show "Powered by Stability AI"; redistributing the weights requires passing on the agreement and a Notice file.
- The licensee owns the generated outputs (as far as law allows) and may use them commercially, but may not use them to train or improve another foundation model. The Acceptable Use Policy applies.
- The T5Gemma encoder adds the Gemma Terms of Use, which permit commercial use subject to their use restrictions (Section 3.2 and the Gemma Prohibited Use Policy).

For a game shipped by a company under USD 1M revenue, that permits using the generated sounds, provided the company registers and credits Stability AI.

### Recommendation

1. The owner decides whether to accept the Stable Audio 3 terms on Hugging Face (`stabilityai/stable-audio-3-small-sfx`) and register for the Community License. That is a human step and was not taken here.
2. If yes, add Stable Audio 3 Small SFX through native nodes: `CheckpointLoaderSimple`, `CLIPLoader` (type `stable_audio`) with `t5gemma_b_b_ul2.safetensors`, `EmptyLatentAudio`, `KSampler` (8 steps, `lcm`, cfg 1 as in the SA3 template; to be checked for the small model), `VAEDecodeAudio`, `SaveAudio`. The weights are small enough to stay well inside free VRAM. Pin the files below (sizes and SHA256 from the Hugging Face API; the upstream gated repo would need a token at rebuild, so it is the owner's call whether to pin that or the Comfy-Org rehost), add an `audio` group to `download_models.py` and `install.sh --only`, and expose a `generate_sound` tool with WAV output and an optional crossfaded loop.
3. If no, the fallback is EzAudio (MIT), which needs a decision to allow non-native inference code; or ACE-Step 1.5 for ambient drones only.

Files that would be pinned (revision `75aad8836271cb5a5123fb58304fae5d3bb4b0f9` of `Comfy-Org/stable-audio-3`; not downloaded):

| Target | Size (bytes) | SHA256 |
|---|---|---|
| `checkpoints/stable_audio_3_small_sfx.safetensors` | 2270384940 | `ed9cf1b6172f1a8c2921a9560c21109ff3239524563ced9dce6dcdef41e2f515` |
| `text_encoders/t5gemma_b_b_ul2.safetensors` | 1187264003 | `1e1eba25be8872edb0d3c6335c6658fd6388e7b14b60da6e454e404cfcd8150e` |

Sources: https://huggingface.co/Comfy-Org/stable-audio-3 , https://huggingface.co/stabilityai/stable-audio-3-small-sfx , https://stability.ai/community-license-agreement , https://ai.google.dev/gemma/terms , https://docs.comfy.org/tutorials/audio/stable-audio/stable-audio-3 , https://huggingface.co/OpenSound/EzAudio , https://ai.sony/blog/introducing-woosh-sony-ais-sound-effect-foundation-model , and the Hugging Face API license tags of each repo above (`cvssp/audioldm2`, `declare-lab/tango2`, `HKUSTAudio/AudioX`, `hkchengrex/MMAudio`, `facebook/audiogen-medium`, `Comfy-Org/YuE2`, `Comfy-Org/ACE-Step-1.5`, `ACE-Step/Ace-Step1.5`, `tencent/HunyuanVideo-Foley`, `FunAudioLLM/ThinkSound`).

## Text to speech (added 2026-11-15)

Use: dialogue for the game *A Nameless Rock: Voices in the Dark*, mainly the ship AI PATHFINDER, including "low power" lines where it struggles to speak. Criteria, in order: weights and code both licensed for commercial use, ungated (no login), runs on the CPU (the GPU is shared and usually busy with ComfyUI), good English quality, small. Voice cloning was a nice to have, not a requirement.

Licences were read from each model's Hugging Face card metadata (`cardData.license`, `gated`) through the Hugging Face API on 2026-11-15, and from the project repositories. This is a reading of the stated licences, not legal advice.

| Candidate | Size | Weights licence | Code licence | Notes | Decision |
|---|---|---|---|---|---|
| **Kokoro-82M v1.0** (hexgrad) | 82M, 327 MB | Apache 2.0, ungated | Apache 2.0 (`kokoro`, `misaki`) | StyleTTS 2 based, 54 voices (28 English), fast on a CPU, word timestamps | **Chosen** |
| Chatterbox, Chatterbox Turbo, Multilingual (Resemble AI) | 0.5B, 350M | MIT, ungated | MIT | Zero shot voice cloning and an exaggeration control. Every output carries Resemble's PerTh neural watermark. The README says that as of August 2026 it is "retired research and is no longer maintained" | Not chosen: unmaintained, slow on a CPU, watermarked output. The upgrade path if cloning one reference voice for PATHFINDER is wanted |
| Kitten TTS mini 0.8 / nano (KittenML) | 80M / 15M | Apache 2.0, ungated | Apache 2.0 | CPU first, a handful of voices. The project has since moved on to a larger KittenTTS 2 with voice cloning (not reviewed) | Not chosen: fewer voices; a reasonable CPU alternative (not compared by listening) |
| Phonon 31m English (Gradium, released 2026-10-20) | 31M, 65 MB q8 GGUF | Apache 2.0, ungated (the 31m English model only; the 40m models are CC BY-NC 4.0) | runtime `ptts` | Built on Kyutai's Pocket TTS and the Mimi codec; streams, CPU friendly; no cloning in the open weights | Not chosen: a month old, new runtime. Worth a listening test later |
| Piper (Rhasspy, now OHF-Voice `piper1-gpl`) | 15 to 60 MB per voice | per voice: each voice's model card names its dataset licence, and some datasets are non-commercial | the original MIT repo is archived; development continues as GPL-3.0 `piper1-gpl`, which links espeak-ng | Very fast, lower quality | Not chosen: each voice needs its own licence check, and quality is below Kokoro |
| Qwen3-TTS 12Hz 1.7B (Alibaba Qwen) | 1.7B | Apache 2.0, ungated | Apache 2.0 | Voice cloning and voice design, multilingual | Not chosen: sized for a GPU; on the shared card it would compete with ComfyUI. A candidate if cloning on the GPU is wanted |
| VoxCPM 1.5 / VoxCPM2 (OpenBMB) | 0.8B / 2B | Apache 2.0, ungated | Apache 2.0 | Cloning, 48 kHz | Not chosen: GPU sized |
| CosyVoice 2 / Fun-CosyVoice 3 0.5B (FunAudioLLM) | 0.5B | Apache 2.0, ungated | Apache 2.0 | Cloning, multilingual | Not chosen: heavier stack, GPU oriented |
| Dia 1.6B, Dia2 2B (Nari Labs) | 1.6B / 2B | Apache 2.0, ungated | Apache 2.0 | Dialogue with non-verbal sounds | Not chosen: GPU sized |
| Zonos v0.1 (Zyphra) | 1.6B | Apache 2.0, ungated | Apache 2.0 | Uses espeak-ng; the hybrid model needs CUDA kernels | Not chosen: GPU sized |
| Kyutai TTS 1.6B | 1.6B | CC BY 4.0, ungated (commercial use allowed with attribution) | Apache 2.0 | Streaming | Not chosen: GPU sized; attribution required |
| Kyutai Pocket TTS | small | CC BY 4.0 | MIT | CPU friendly | Excluded: gated on Hugging Face (needs a login) |
| Orpheus 3B (Canopy Labs) | 3B | card says Apache 2.0, but it is a Llama 3.2 fine tune, so the Llama 3.2 Community License also applies; gated | Apache 2.0 | | Excluded: gated, Llama licence, GPU sized |
| Sesame CSM-1B | 1B | Apache 2.0 card, Llama backbone; gated | Apache 2.0 | | Excluded: gated |
| NeuTTS Air (Neuphonic) | 0.7B | Apache 2.0; gated | Apache 2.0 | CPU cloning | Excluded: gated |
| VibeVoice 1.5B / Realtime 0.5B (Microsoft) | 1.5B / 0.5B | MIT, ungated | MIT | Long form, multi speaker | Not chosen: GPU oriented, aimed at podcasts |
| F5-TTS | 0.3B | CC BY-NC 4.0 | MIT | | Excluded: non-commercial weights |
| XTTS v2 (Coqui) | 0.5B | Coqui Public Model License (non-commercial); Coqui has shut down, so no commercial licence can be bought | MPL 2.0 | | Excluded |
| Voxtral 4B TTS (Mistral, 2026-03) | 4B | CC BY-NC 4.0 | | | Excluded: non-commercial |
| IndexTTS-2 (bilibili) | | custom licence (none in the card metadata) | | | Excluded: licence not clear |
| OpenAudio S1 mini (Fish Audio), Higgs Audio v2 (Boson AI) | | the API refused anonymous reads (gated); Fish Audio's weights are CC BY-NC-SA | | | Excluded |

Decision: **Kokoro-82M v1.0**, revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, the model, `config.json` and the 28 English voices, 341.9 MB in all, pinned by SHA256 in `setup/tts.manifest`. It is the only candidate that is both clearly licensed for commercial use and ungated, small enough to run on the CPU in a few seconds per line, and mature (its packages have been stable since early 2025). The trade off is no voice cloning: PATHFINDER's voice is one of the stock voices or an average of several, and its character comes from the effects.

### Dependencies of the chosen stack

The text to speech venv (`setup/tts-requirements.lock`, 1.3 GB) holds, among others:

| Package | Licence | How it is used |
|---|---|---|
| `kokoro` 0.9.4, `misaki` 0.9.4 | Apache 2.0 | Model and English G2P (grapheme to phoneme) |
| `torch` 2.14.1+cpu | BSD style (Apache 2.0, BSD, MIT parts) | CPU inference; the CPU only build cannot touch the GPU |
| `transformers`, `huggingface-hub` | Apache 2.0 | Model code; no downloads at runtime (`HF_HUB_OFFLINE=1`, weights read from `<root>/models/tts`) |
| `spacy` 3.8, `en_core_web_sm` 3.8.0 | MIT | Part of speech tags for misaki; installed from a pinned wheel so misaki never downloads it |
| `num2words` | LGPL 2.1 | Numbers to words ("forty"), imported as an unmodified Python library |
| `phonemizer-fork` 3.3.2 | GPL 3.0 or later | Fallback G2P for words missing from misaki's dictionary |
| `espeakng-loader` 0.2.4 | MIT wrapper around a bundled `libespeak-ng.so` (espeak-ng, GPL 3.0) | The phonemizer loads this shared library into the runner process with ctypes |

On the GPL parts, plainly: espeak-ng and `phonemizer-fork` are GPL 3.0 and are loaded as libraries into the runner process (a linked use, not a separate program). The GPL governs copying and distributing that software. It does not cover what the program outputs: the generated speech is not a copy or derivative of espeak-ng (it is phonemes turned into audio by an Apache 2.0 model), so the WAV and OGG files can be used in a commercial game without GPL obligations. The obligations would apply only if the text to speech runtime itself were shipped to others (for example bundled in the game to synthesise speech on players' machines): then the combined runtime would have to be distributed under GPL terms. This repository ships only its own MIT source and installs the GPL packages from PyPI on the user's machine. If an all permissive runtime is ever needed, misaki can run without the espeak fallback (`fallback=None`); words outside its dictionary are then skipped.

The Apache 2.0 licence of the Kokoro weights asks for its notice to be kept when the weights are redistributed; generated audio needs no attribution. Kokoro's training data is described by its card as permissive or non-copyrighted audio plus synthetic audio from closed TTS providers; nothing in the licence restricts using the output.

### Effects

Effects are ffmpeg filter graphs built in `src/localgen_mcp/tts/effects.py` (ffmpeg is a separate program; LGPL or GPL depending on the build, used as a command line tool, which places no terms on the audio). Without ffmpeg the runner renders numpy approximations. `glitch` uses Kokoro's word timestamps so stutters land on word onsets.

### Text to speech sources

- Kokoro-82M card and voice grades: https://huggingface.co/hexgrad/Kokoro-82M and its `VOICES.md`; code: https://github.com/hexgrad/kokoro , https://github.com/hexgrad/misaki
- Chatterbox (archive notice, PerTh watermark): https://github.com/resemble-ai/chatterbox
- Kitten TTS: https://huggingface.co/KittenML/kitten-tts-mini-0.8
- Phonon: https://huggingface.co/gradium/phonon-31m-en (family table and licences)
- Piper: https://github.com/rhasspy/piper (archived) and https://github.com/OHF-Voice/piper1-gpl (GPL-3.0); voices: https://huggingface.co/rhasspy/piper-voices
- espeak-ng (GPL-3.0): https://github.com/espeak-ng/espeak-ng ; espeakng-loader (MIT): https://github.com/thewh1teagle/espeakng-loader
- Other cards, read through `https://huggingface.co/api/models/<repo>`: Qwen/Qwen3-TTS-12Hz-1.7B-Base, openbmb/VoxCPM2, FunAudioLLM/Fun-CosyVoice3-0.5B-2512, nari-labs/Dia2-2B, Zyphra/Zonos-v0.1-transformer, kyutai/tts-1.6b-en_fr, kyutai/pocket-tts, canopylabs/orpheus-3b-0.1-ft, sesame/csm-1b, neuphonic/neutts-air, microsoft/VibeVoice-1.5B, SWivid/F5-TTS, coqui/XTTS-v2, mistralai/Voxtral-4B-TTS-2603, IndexTeam/IndexTTS-2
- Survey articles used to find candidates (secondary): https://findskill.ai/blog/best-open-source-tts-2026/ , https://efeanidi.com/articles/phonon-tts/

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
