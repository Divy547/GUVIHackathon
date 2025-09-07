# Full app: per-scene genres + per-scene regenerate + generate images for all scenes
import streamlit as st
import zipfile
import io
import re
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw
from transformers import pipeline
import torch
from fpdf import FPDF
import io


import tempfile

def create_story_pdf_with_images(scenes, title="My Story"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(10)

    for i, scene in enumerate(scenes):
        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, f"Scene {i+1}: {scene.title}")
        pdf.ln(2)

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, scene.text)
        pdf.ln(3)

        if scene.image:
            # Save PIL image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                scene.image.save(tmp_file.name, format="PNG")
                pdf.image(tmp_file.name, w=pdf.w - 30)
            pdf.ln(10)

    # Get PDF as byte string
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)




# NOTE: SD-Turbo (diffusers) section is optional; placeholder mode works without heavy downloads.
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False


# --------------------------
# Data structure
# --------------------------
@dataclass
class Scene:
    title: str
    text: str
    prompt: str
    image: Optional[Image.Image] = None


# --------------------------
# Constants / choices
# --------------------------
GENRES = ["Fantasy", "Sci-Fi", "Mystery", "Comedy",
          "Drama", "Adventure", "Horror", "Slice of Life"]
TONES = ["Serious", "Lighthearted", "Epic", "Dark", "Hopeful"]
AUDIENCES = ["Kids", "Teens", "Adults"]
ART_STYLES = ["Digital Painting", "Anime",
              "Cartoon", "Watercolor", "Realistic"]


# --------------------------
# Model loaders (cached)
# --------------------------
@st.cache_resource
def load_text_model(model_name: str = "google/flan-t5-base"):
    """Load a text2text model pipeline (FLAN-T5 by default)."""
    return pipeline("text2text-generation", model=model_name)


@st.cache_resource
def load_sd_turbo():
    """Load SD-Turbo via diffusers if available. This may download large weights."""
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("diffusers not available in environment.")
    model_id = "stabilityai/sd-turbo"
    # use float16 if cuda available
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


# --------------------------
# Generation helpers
# --------------------------
def generate_scenes(pipe, idea, global_genre, tone, audience, n_scenes, art_style):
    """Ask the model to produce EXACTLY n_scenes in a strict format; parse with regex."""
    prompt = f"""
You are a professional story writer.
Expand the following story idea into EXACTLY {n_scenes} scenes.

Each scene MUST follow this format exactly:

Scene <number>:
Title: <short title>
Narrative: <3-5 sentences of story text for {audience}>
Image Prompt: <detailed visual description suitable for {art_style} illustration>

Story idea: {idea}
Default genre: {global_genre}
Tone: {tone}
"""
    out = pipe(prompt, max_new_tokens=1200, temperature=0.7, do_sample=True)
    text = out[0].get("generated_text", "") if isinstance(
        out, list) else str(out)

    # Regex to capture scenes robustly
    pattern = r"Scene\s*\d+:\s*Title:\s*(.*?)\s*Narrative:\s*(.*?)\s*Image Prompt:\s*(.*?)(?=Scene\s*\d+:|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    scenes: List[Scene] = []
    for title, narrative, img_prompt in matches:
        title = title.strip()
        narrative = narrative.strip()
        img_prompt = img_prompt.strip()
        if not img_prompt and narrative:
            img_prompt = f"Illustration in {art_style} style of: {narrative}"
        if title and narrative:
            scenes.append(
                Scene(title=title, text=narrative, prompt=img_prompt))

    # fallback: if parsing failed, create n_scenes by heuristics using the raw text
    if not scenes:
        # simple split by paragraphs/sentences to attempt to create n scenes
        chunks = re.split(r"\n\n+|\.\s+", text)
        chunks = [c.strip() for c in chunks if c.strip()]
        for i in range(n_scenes):
            txt = chunks[i] if i < len(chunks) else (
                chunks[-1] if chunks else text)
            title = f"Scene {i+1}"
            prompt_text = f"Illustration in {art_style} style of: {txt}"
            scenes.append(Scene(title=title, text=txt, prompt=prompt_text))

    # ensure exact length
    if len(scenes) > n_scenes:
        scenes = scenes[:n_scenes]
    while len(scenes) < n_scenes:
        idx = len(scenes) + 1
        scenes.append(Scene(title=f"Scene {idx}", text=f"A moment {idx} in the story.",
                      prompt=f"Illustration in {art_style} style of a key moment {idx}."))

    return scenes


def generate_single_scene(pipe, idea, scene_genre, tone, audience, art_style, scene_index):
    """Generate a single scene (used for per-scene regenerate)."""
    prompt = f"""
You are a story writer. Produce one scene (Scene {scene_index}) for the idea below.
Return exactly three lines prefixed as:
Title: ...
Narrative: ...
Image Prompt: ...

Story idea: {idea}
Genre: {scene_genre}
Tone: {tone}
Audience: {audience}
Art style: {art_style}
"""
    out = pipe(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)
    text = out[0].get("generated_text", "") if isinstance(
        out, list) else str(out)

    # Simple parse for the 3 fields
    title = ""
    narrative = ""
    img_prompt = ""
    for line in text.splitlines():
        if line.strip().lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("narrative:"):
            narrative = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("image prompt:"):
            img_prompt = line.split(":", 1)[1].strip()

    if not img_prompt and narrative:
        img_prompt = f"Illustration in {art_style} style of: {narrative}"

    # fallback if nothing parsed
    if not title and not narrative:
        title = f"Scene {scene_index}"
        narrative = text.strip()
        img_prompt = f"Illustration in {art_style} style of: {narrative}"

    return Scene(title=title, text=narrative, prompt=img_prompt)


# --------------------------
# Image helpers
# --------------------------
def placeholder_image(prompt: str, size=(512, 512)):
    """Simple placeholder image with the prompt text drawn for quick testing."""
    img = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    lines = []
    # wrap long prompt into lines
    for i in range(0, len(prompt), 60):
        lines.append(prompt[i:i+60])
    draw.text((8, 8), "\n".join(lines[:10]), fill=(20, 20, 20))
    return img


def render_image_sd_turbo(pipe, prompt: str, seed: int, width: int, height: int, num_steps: int):
    """Render using SD-Turbo pipeline (if loaded)."""
    # generator creation depending on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(prompt=prompt, width=width, height=height,
               num_inference_steps=num_steps, guidance_scale=0.0, generator=gen)
    return out.images[0]


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(
    page_title="Per-Scene Genres Story Generator", layout="wide")
st.title("📖 Per-Scene Genres — Story Generator")

# Sidebar - global defaults
with st.sidebar:
    st.header("Global Settings (defaults)")
    idea = st.text_area(
        "Story idea", value="A young girl finds a secret door in her grandmother's attic.", height=120)
    global_genre = st.selectbox(
        "Default genre (applies to all scenes initially)", GENRES, index=0)
    tone = st.selectbox("Tone", TONES, index=0)
    audience = st.selectbox("Target audience", AUDIENCES, index=0)
    art_style = st.selectbox("Art style", ART_STYLES, index=0)
    n_scenes = st.slider("Number of scenes", min_value=3, max_value=6,
                         value=5, help="Set to 5 if you want 5 scenes.")
    st.markdown("---")
    st.header("Image Settings")
    image_engine = st.selectbox("Image engine", ["Placeholder (no model download)"] + (
        ["Stable Diffusion Turbo"] if DIFFUSERS_AVAILABLE else []))
    sd_steps = st.slider("SD Turbo steps (if using SD)", 1, 30, 20)
    width = st.select_slider("Image width", options=[
                             256, 384, 512, 640, 768], value=512)
    height = st.select_slider("Image height", options=[
                              256, 384, 512, 640, 768], value=512)
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    st.markdown("---")
    st.caption("Tip: change per-scene genre below after generating the narrative, then press 'Regenerate Scene' for that scene to rewrite it in the chosen genre.")


# Buttons (main area)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    gen_btn = st.button("✍️ Generate Narrative (use default genre)")
with col2:
    regen_all_btn = st.button(
        "🔁 Regenerate All Scenes (use each scene's genre)")
with col3:
    img_btn = st.button("🎨 Generate Images for All Scenes")


# Initialize session state lists
if "scenes" not in st.session_state:
    st.session_state["scenes"] = []
if "scene_genres" not in st.session_state:
    st.session_state["scene_genres"] = []
if "scene_tones" not in st.session_state:
    st.session_state["scene_tones"] = []


# Generate full narrative (initial)
if gen_btn:
    if not idea.strip():
        st.warning("Enter a story idea first.")
    else:
        with st.spinner("Generating scenes..."):
            pipe = load_text_model()
            scenes = generate_scenes(
                pipe, idea, global_genre, tone, audience, n_scenes, art_style)
            st.session_state["scenes"] = scenes
            # initialize per-scene genre & tone with global defaults (resize if needed)
            st.session_state["scene_genres"] = [
                global_genre for _ in range(len(scenes))]
            st.session_state["scene_tones"] = [
                tone for _ in range(len(scenes))]
        st.success("Narrative generated. Edit scenes and per-scene genres below.")


# Helper to ensure lists match n_scenes
def ensure_lengths():
    # ensure scenes length equals n_scenes (if user changed slider later)
    s_len = len(st.session_state["scenes"])
    if s_len != n_scenes:
        # if fewer, pad; if more, trim
        if s_len < n_scenes:
            for i in range(s_len, n_scenes):
                st.session_state["scenes"].append(
                    Scene(title=f"Scene {i+1}", text="(empty)", prompt="(edit prompt)"))
        else:
            st.session_state["scenes"] = st.session_state["scenes"][:n_scenes]
    # scene_genres
    if "scene_genres" not in st.session_state or len(st.session_state["scene_genres"]) != n_scenes:
        st.session_state["scene_genres"] = [
            global_genre for _ in range(n_scenes)]
    if "scene_tones" not in st.session_state or len(st.session_state["scene_tones"]) != n_scenes:
        st.session_state["scene_tones"] = [tone for _ in range(n_scenes)]


ensure_lengths()
scenes: List[Scene] = st.session_state["scenes"]


# Editable scene list with per-scene genre and regenerate button
if scenes:
    st.subheader("✍️ Edit Scenes (per-scene genre enabled)")

    updated_scenes: List[Scene] = []
    for i, scene in enumerate(scenes, start=1):
        idx = i - 1
        with st.expander(f"{i}. {scene.title}", expanded=True):
            # Per-scene genre selector (defaults from session_state or global)
            current_genre = st.session_state["scene_genres"][idx] if idx < len(
                st.session_state["scene_genres"]) else global_genre
            # safe index for selectbox initial selection
            try:
                init_idx = GENRES.index(current_genre)
            except ValueError:
                init_idx = 0
            new_genre = st.selectbox(
                "Scene Genre", options=GENRES, index=init_idx, key=f"scene_genre_{i}")
            st.session_state["scene_genres"][idx] = new_genre

            # Optional per-scene tone (kept simple)
            current_tone = st.session_state["scene_tones"][idx] if idx < len(
                st.session_state["scene_tones"]) else tone
            try:
                tone_init = TONES.index(current_tone)
            except ValueError:
                tone_init = 0
            new_tone = st.selectbox(
                "Scene Tone", options=TONES, index=tone_init, key=f"scene_tone_{i}")
            st.session_state["scene_tones"][idx] = new_tone

            # Editable fields
            new_title = st.text_input(
                "Title", value=scene.title, key=f"title_{i}")
            new_text = st.text_area(
                "Narrative", value=scene.text, key=f"text_{i}", height=160)
            new_prompt = st.text_area(
                "Image Prompt", value=scene.prompt, key=f"prompt_{i}", height=120)

            # Regenerate this scene button (uses per-scene genre)
            regen_key = f"regen_scene_{i}"
            if st.button(f"🔁 Regenerate Scene {i} (use this scene's genre)", key=regen_key):
                with st.spinner(f"Regenerating scene {i} in genre '{new_genre}'..."):
                    pipe = load_text_model()
                    new_scene = generate_single_scene(
                        pipe, idea, new_genre, new_tone, audience, art_style, i)
                    # overwrite the fields for this scene
                    new_title = new_scene.title
                    new_text = new_scene.text
                    new_prompt = new_scene.prompt
                    scene.image = None  # clear old image because prompt changed
                st.success(f"Scene {i} regenerated.")

            # Save updated scene
            updated_scenes.append(
                Scene(title=new_title, text=new_text, prompt=new_prompt, image=scene.image))

    st.session_state["scenes"] = updated_scenes
    scenes = st.session_state["scenes"]


# Regenerate all scenes using per-scene genres (button)
if regen_all_btn and scenes:
    with st.spinner("Regenerating all scenes using each scene's selected genre..."):
        pipe = load_text_model()
        for idx in range(len(scenes)):
            sg = st.session_state["scene_genres"][idx]
            st.session_state["scenes"][idx] = generate_single_scene(
                pipe, idea, sg, st.session_state["scene_tones"][idx], audience, art_style, idx + 1)
    st.success("All scenes regenerated using per-scene genres.")


# Generate images for all scenes
# Generate images for all scenes
if img_btn and scenes:
    with st.spinner("Generating images for all scenes..."):
        if image_engine.startswith("Placeholder"):
            for idx, s in enumerate(st.session_state["scenes"]):
                # Use each scene's own prompt here
                st.session_state["scenes"][idx].image = placeholder_image(
                    s.prompt, size=(width, height))
        else:
            try:
                sd_pipe = load_sd_turbo()
                for idx, s in enumerate(st.session_state["scenes"]):
                    # Use each scene's own prompt and unique seed
                    img = render_image_sd_turbo(
                        sd_pipe,
                        s.prompt,                 # <-- use s.prompt
                        seed=int(seed) + idx,     # different seed per scene
                        width=width,
                        height=height,
                        num_steps=sd_steps
                    )
                    st.session_state["scenes"][idx].image = img
            except Exception as e:
                st.error(f"Image generation failed: {e}")
    st.success("Images generated for all scenes.")


# Storybook preview: show per-scene genre as well
if scenes:
    st.header("📖 Storybook Preview (editable)")
    for i, s in enumerate(scenes, start=1):
        st.markdown(
            f"#### {i}. {s.title} — *Genre: {st.session_state['scene_genres'][i-1]}*")
        cols = st.columns([1, 1.4])
        with cols[0]:
            st.write(s.text)
        with cols[1]:
            if s.image:
                st.image(s.image, caption=s.prompt, use_column_width=True)
            else:
                st.info(
                    "No image yet. Click 'Generate Images' to create images for all scenes.")


# Export (ZIP or simple PDF)
if scenes:
    st.markdown("---")
    st.header("📦 Export")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗂️ Export ZIP (images + story.txt)"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                # write story text
                story_txt = ""
                for idx, s in enumerate(st.session_state["scenes"], start=1):
                    story_txt += f"{idx}. {s.title} (Genre: {st.session_state['scene_genres'][idx-1]})\n{s.text}\n\n"
                z.writestr("story.txt", story_txt.encode("utf-8"))
                # add images if present
                for idx, s in enumerate(st.session_state["scenes"], start=1):
                    if s.image:
                        b = io.BytesIO()
                        s.image.save(b, format="PNG")
                        z.writestr(f"scene_{idx:02d}.png", b.getvalue())
            st.download_button("Download ZIP", buf.getvalue(),
                               file_name="story_assets.zip")
    with c2:
        if st.button("📄 Export PDF (with images)"):
            try:
                pdf_file = create_story_pdf_with_images(
                    st.session_state["scenes"], title=idea)
                st.download_button(
                    label="Download Story PDF",
                    data=pdf_file,
                    file_name="storybook.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF export failed: {e}")


st.markdown("---")
st.caption("You can set a genre per scene above. After changing a scene's genre, use 'Regenerate Scene' to rewrite that single scene in the chosen genre, or 'Regenerate All Scenes' to rewrite all using each scene's genre.")
