
# Text‑to‑Image Story Generator (Free Models)

An end‑to‑end Streamlit app that takes a short story idea and generates a multi‑scene narrative with image prompts, then illustrates each scene. Uses only free, open models:

- **Text**: `google/flan-t5-base` (small, no API key needed)
- **Images**: `stabilityai/sd-turbo` via `diffusers` (fast on GPU; offers a placeholder mode for CPU‑only)

## Features
- Genre, tone, audience, art style
- 3–5 scenes with titles, narrative, and image prompts
- Edit scene text and prompts before rendering
- Generate images with SD‑Turbo or create **placeholders** (no model download)
- Export **PDF** or **ZIP** (images + story.txt)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

> **Tip**: On CPU‑only machines, start with **Placeholder** image engine so you can test the flow without large model downloads. Switch to **SD‑Turbo** later (GPU recommended).

## Notes
- First run will download models from Hugging Face.
- You can swap FLAN‑T5 with any local text model supported by `transformers`.
- For different image engines, try `segmind/SSD-1B` or `runwayml/stable-diffusion-v1-5` in the code.
