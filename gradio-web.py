import gradio as gr
import numpy as np
import logging
from kittentts import KittenTTS

# -----------------------
# Logging configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("kitten_tts")

# Enable uvicorn access logs
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

AVAILABLE_MODELS = {
    "kitten-tts-mini": "KittenML/kitten-tts-mini-0.8",
    "kitten-tts-micro": "KittenML/kitten-tts-micro-0.8",
    "kitten-tts-nano": "KittenML/kitten-tts-nano-0.8",
}

AVAILABLE_VOICES = ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']

DEFAULT_TEXT = (
    "Hello and welcome to the Kitten TTS demo. "
    "This example shows how to convert text into natural sounding speech. "
    "You can choose different voices and models from the interface. "
    "Feel free to experiment with your own text."
)

loaded_models = {}

def get_model(model_key):
    if model_key not in loaded_models:
        logger.info(f"Loading model: {model_key}")
        loaded_models[model_key] = KittenTTS(AVAILABLE_MODELS[model_key])
        logger.info(f"Model loaded: {model_key}")
    return loaded_models[model_key]

def generate_audio(text, voice, model_key):
    if not text.strip():
        logger.warning("Empty text input received.")
        return None

    logger.info(f"Generating audio | Model: {model_key} | Voice: {voice}")
    model = get_model(model_key)
    audio = model.generate(text, voice=voice)
    logger.info("Audio generation completed.")

    return (24000, np.array(audio))

with gr.Blocks(title="KittenTTS Demo") as app:
    gr.Markdown("## KittenTTS Text-to-Speech Demo")

    model_dropdown = gr.Dropdown(
        choices=list(AVAILABLE_MODELS.keys()),
        value="kitten-tts-mini",
        label="Select Model"
    )

    text_input = gr.Textbox(
        label="Enter Text",
        value=DEFAULT_TEXT,
        lines=5
    )

    voice_dropdown = gr.Dropdown(
        choices=AVAILABLE_VOICES,
        value="Jasper",
        label="Select Voice"
    )

    generate_button = gr.Button("Generate Speech")

    audio_output = gr.Audio(
        label="Generated Audio",
        type="numpy"
    )

    generate_button.click(
        fn=generate_audio,
        inputs=[text_input, voice_dropdown, model_dropdown],
        outputs=audio_output
    )

if __name__ == "__main__":
    logger.info("Starting Gradio server on 0.0.0.0:7860")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
