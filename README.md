import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import os

# ---------------------------
# Load models
# ---------------------------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load Stable Diffusion
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    )
    sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(device)

    # Load BLIP captioning model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device).eval()

    return sd_pipe, blip_processor, blip_model, device


# ---------------------------
# Prompt enrichment
# ---------------------------
def enrich_prompt(prompt):
    return f"A highly detailed, photorealistic image of {prompt}, 4k, vibrant lighting, cinematic atmosphere"


# ---------------------------
# Image generation
# ---------------------------
def generate_image(prompt, sd_pipe):
    enriched_prompt = enrich_prompt(prompt)
    negative_prompt = (
        "blurry, low resolution, distorted, bad anatomy, extra limbs, poorly drawn face, disfigured, mutated"
    )
    image = sd_pipe(
        enriched_prompt,
        num_inference_steps=50,
        guidance_scale=9.0,
        negative_prompt=negative_prompt
    ).images[0]
    return image


# ---------------------------
# Caption generation
# ---------------------------
def generate_caption(image, blip_processor, blip_model, device):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


# ---------------------------
# Full pipeline: prompt → image + caption
# ---------------------------
def full_pipeline(prompt):
    image = generate_image(prompt, sd_pipe)
    caption = generate_caption(image, blip_processor, blip_model, device)
    return image, caption


# ---------------------------
# Wrapper for uploaded image captioning
# ---------------------------
def caption_wrapper(image):
    return generate_caption(image, blip_processor, blip_model, device)


# ---------------------------
# Initialize models
# ---------------------------
sd_pipe, blip_processor, blip_model, device = load_models()


# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🖼 VisionVerse: Image Generator & Captioning App")
    gr.Markdown("Generate realistic images from text or describe uploaded images.")

    with gr.Tabs():
        with gr.TabItem("1️⃣ Text → Image"):
            with gr.Row():
                prompt_input = gr.Textbox(label="Enter a prompt", placeholder="e.g. A cat astronaut exploring Mars")
                generate_btn = gr.Button("Generate")
            image_output = gr.Image(label="Generated Image", interactive=True)
            caption_output = gr.Textbox(label="Caption")
            generate_btn.click(fn=full_pipeline, inputs=prompt_input, outputs=[image_output, caption_output])

        with gr.TabItem("2️⃣ Image → Caption"):
            image_input = gr.Image(type="pil", label="Upload an image")
            caption_btn = gr.Button("Generate Caption")
            caption_result = gr.Textbox(label="Generated Caption")
            caption_btn.click(fn=caption_wrapper, inputs=image_input, outputs=caption_result)


# ---------------------------
# Launch app (Render-friendly)
# ---------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
