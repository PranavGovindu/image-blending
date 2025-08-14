#  Image Composer – Developer Guide

## What this app does
- Takes a single text prompt from the browser form.
- Uses a local Ollama Qwen model to generate:
  - Short story
  - Character description (for image prompting)
  - Background description (for image prompting)
- Generates two images with Hugging Face Inference (SDXL):
  - Character image (foreground, then alpha-matted)
  - Background image (scene)
- Composites character onto the background with advanced blending.
- Writes outputs to `media/` and renders them in the UI.

Outputs written to `media/`:
- `character_raw.png` (RGB from SDXL)
- `character_rgba.png` (alpha-matted)
- `background.png` (scene)
- `combined.png` (final composite)

## System requirements
- Python 3.11+
- `uv` (for isolated, reproducible Python env)
- Ollama running locally
- Valid Hugging Face access token (for SDXL)

## One-time install

1) Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

2) Install project deps
```bash
cd /home/pranni/iiscc
uv sync
```

3) Install and start Ollama
- Download from `https://ollama.com/download`
- Pull the model and start the server:
```bash
ollama pull qwen2:1.5b-instruct
ollama serve
```

## Environment variables
Set these in the same shell you run the server from.

```bash
# Ollama (local LLM)
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2:1.5b-instruct

# Hugging Face (image generation via SDXL)
export HF_TOKEN=hf_...your_token...
```

Notes:
- If `HF_TOKEN` is missing or invalid, SDXL responses will fail (401). Fix the token.
- Defaults: `OLLAMA_URL=http://localhost:11434`, `OLLAMA_MODEL=qwen2:1.5b-instruct`.

## Run the app
```bash
uv run python manage.py runserver 0.0.0.0:8000
```
Open `http://localhost:8000`, enter a prompt, click Generate.

This project is stateless; no database is used.

## How the pipeline works

- File: `creative/pipeline.py`

1) Prompt ingestion
   - `creative/views.py` collects the text prompt and calls `orchestrate()`.

2) Text generation (Ollama → Qwen)
   - `generate_texts()` sends a single instruction to Ollama.
   - Expects strict JSON keys: `story`, `character_description`, `background_description`.

3) Prompt engineering
   - `craft_character_prompt()` and `craft_background_prompt()` translate the descriptions
     into image-friendly prompts with composition/lighting constraints.

4) Image generation (Hugging Face Inference, SDXL)
   - `generate_character_image()` → 1024×1024
   - `generate_background_image()` → 1536×1024
   - Requires `HF_TOKEN`.

5) Compositing
   - `combine_images()` performs:
     - LAB color transfer for palette harmony
     - Edge refinement and soft feathering
     - Rim light derived from background sample
     - Contact shadow estimation from background light direction
     - Subtle atmospheric perspective, grain, micro aberration

6) UI
   - Renders story, descriptions, and final composite; assets saved under `media/`.

## Camera/angle control (prompting tips)
Add these phrases to your input prompt to control framing:
- Low-angle hero shot
- High-angle top-down view / bird’s-eye view
- Worm’s-eye view
- Eye-level, 35mm lens
- Over-the-shoulder, 85mm portrait
- Dutch tilt, dramatic angle

## Troubleshooting
- 401 from SDXL (HF):
  - Symptom: server logs show `HF Response status: 401` and JSON error.
  - Fix: `export HF_TOKEN=...` in the same shell, restart server.
- Ollama connection refused:
  - Start with `ollama serve` and ensure `OLLAMA_URL` matches.
- Image looks pasted or mismatched:
  - Add angle/lighting language to your prompt (see tips above).
  - The compositor expects a subject clearly separated from the background.

## Future improvements
- Prompt templates
  - Add camera metadata controls (focal length, sensor size, aperture) per scene type.
  - Style presets (anime, cinematic noir, watercolor, photoreal) toggled via UI.
  - Automatic time-of-day/lighting extraction from user prompt with validation.

- Robust local image generation
  - Replace HF SDXL with a local pipeline (e.g., diffusers, ComfyUI, or Automatic1111 API)
    to remove external dependencies and tokens.

- Better compositing
  - Monocular depth estimation to drive depth-aware feathering and haze.
  - Soft body/cloth-aware matting for hair/fabric edges.
  - Ground-plane detection to place contact shadows with perspective.

- UX
  - Progress indicator with stages (LLM → BG → CHAR → COMPOSE).
  - Download bundle (story + prompts + images) as a zip.

## Minimal file map
```
iiscc/
├─ creative/
│  ├─ forms.py
│  ├─ pipeline.py        # all orchestration, gen, compositing
│  ├─ templates/
│  ├─ urls.py
│  └─ views.py
├─ iiscc_site/
│  ├─ settings.py        # minimal Django config (no DB/admin)
│  └─ urls.py
├─ manage.py
├─ media/                # outputs
├─ pyproject.toml
└─ DOCUMENTATION.md
```

