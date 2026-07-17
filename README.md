# 🖼️ VisionVerse: AI-Powered Image Generation & Captioning

VisionVerse is a multimodal Generative AI application that combines **Text-to-Image Generation** and **Image Captioning** into a single interactive web application. Built using **Stable Diffusion v1.5**, **BLIP**, **PyTorch**, and **Gradio**, the application enables users to generate realistic images from natural language prompts and automatically generate descriptive captions for uploaded or generated images.

---

## 🚀 Features

- 🎨 **Text-to-Image Generation**
  - Generate high-quality, photorealistic images using Stable Diffusion v1.5.
  - Prompt enrichment for enhanced image quality.
  - Negative prompting to reduce visual artifacts.
  - Configurable inference parameters.

- 🖼️ **Image Captioning**
  - Upload an image and generate an AI-powered descriptive caption.
  - Powered by Salesforce BLIP.

- 🔄 **End-to-End AI Pipeline**
  - Generate an image from a text prompt.
  - Automatically caption the generated image.
  - Demonstrates a complete multimodal AI workflow.

- 💻 **Interactive Web Interface**
  - Built using Gradio.
  - User-friendly interface with dedicated tabs for each task.

---

# 📌 Project Motivation

Recent advances in Generative AI have enabled powerful models for both image generation and image understanding. However, these capabilities are often available as separate applications.

VisionVerse integrates these technologies into a unified platform, allowing users to seamlessly transition between generating images and understanding visual content.

---

# 🏗️ System Architecture

```
                    User
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Text Prompt             Upload Image
          │                       │
          ▼                       ▼
 Stable Diffusion           BLIP Captioning
(Image Generation)              Model
          │                       │
          ▼                       ▼
 Generated Image          Generated Caption
          │
          ▼
       BLIP Model
          │
          ▼
 Caption for Generated Image
          │
          ▼
       Gradio Interface
```

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Image Generation | Stable Diffusion v1.5 |
| Diffusion Library | Hugging Face Diffusers |
| Scheduler | EulerDiscreteScheduler |
| Image Captioning | Salesforce BLIP |
| Transformers | Hugging Face Transformers |
| Frontend | Gradio |
| Deployment | Render / Local |

---

# 🤖 Models Used

## Stable Diffusion v1.5

**Model**

```
runwayml/stable-diffusion-v1-5
```

**Purpose**

Generates high-quality images from textual prompts using latent diffusion.

**Components**

- CLIP Text Encoder
- U-Net
- Variational Autoencoder (VAE)
- Euler Scheduler

---

## BLIP (Bootstrapping Language-Image Pretraining)

**Model**

```
Salesforce/blip-image-captioning-base
```

**Purpose**

Generates natural language captions from input images by combining computer vision and transformer-based language generation.

---

# 🔄 Workflow

## Text → Image → Caption

```
User Prompt
      │
      ▼
Prompt Enrichment
      │
      ▼
Stable Diffusion
      │
      ▼
Generated Image
      │
      ▼
BLIP Captioning
      │
      ▼
Generated Caption
```

---

## Image → Caption

```
Uploaded Image
       │
       ▼
BLIP Processor
       │
       ▼
BLIP Model
       │
       ▼
Generated Caption
```

---

# ✨ Prompt Engineering

The application enriches user prompts before image generation to improve realism and detail.

### Example

**Input**

```
A tiger in a forest
```

**Enhanced Prompt**

```
A highly detailed, photorealistic image of a tiger in a forest,
4k, vibrant lighting, cinematic atmosphere
```

This improves:

- Image realism
- Lighting
- Detail
- Overall visual quality

---

# 🚫 Negative Prompting

To minimize unwanted artifacts, the following negative prompt is used:

```
blurry,
low resolution,
distorted,
bad anatomy,
extra limbs,
poorly drawn face,
disfigured,
mutated
```

This guides the diffusion model away from generating undesirable outputs.

---

# ⚙️ Inference Configuration

| Parameter | Value |
|-----------|------:|
| Scheduler | EulerDiscreteScheduler |
| Inference Steps | 50 |
| Guidance Scale | 9.0 |
| Precision | FP16 (GPU) / FP32 (CPU) |

---

# 📂 Project Structure

```
VisionVerse/
│
├── app.py
├── requirements.txt
├── README.md
├── assets/
└── examples/
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/visionverse.git
cd visionverse
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
python app.py
```

The application will launch locally at

```
http://localhost:7860
```

---

# 📸 Example Usage

## Text-to-Image

**Input**

```
A futuristic city at sunset
```

**Output**

- AI-generated image
- Automatically generated caption

---

## Image Captioning

Upload an image of a dog playing in snow.

**Generated Caption**

```
A brown dog running through snow.
```

---

# ⚡ Performance Optimizations

- GPU acceleration using CUDA
- Mixed precision (FP16) inference on GPU
- Models loaded only once during startup
- Inference mode using `.eval()`
- Disabled gradient computation using `torch.no_grad()`

---

# 🚀 Future Enhancements

- Image-to-Image Generation
- Stable Diffusion XL (SDXL)
- LoRA Fine-Tuned Models
- Prompt History
- User Authentication
- Download & Share Images
- Multiple Diffusion Schedulers
- Batch Image Generation
- Image Editing & Inpainting
- Cloud Deployment with Scalable Inference

---

# 📚 Learning Outcomes

This project provided hands-on experience with:

- Generative AI
- Diffusion Models
- Hugging Face Diffusers
- Vision-Language Models
- Prompt Engineering
- Image Captioning
- PyTorch Inference
- GPU Acceleration
- Gradio Interface Development
- Multimodal AI Systems

---

# 🌍 Applications

VisionVerse can be applied in:

- Digital Content Creation
- Graphic Design
- Marketing & Advertising
- Education
- Accessibility
- E-commerce
- Social Media Content Generation
- Creative AI Research

---

# 🙏 Acknowledgements

- Hugging Face Diffusers
- Hugging Face Transformers
- Stable Diffusion v1.5 (RunwayML)
- Salesforce BLIP
- PyTorch
- Gradio

---

# 📄 License

This project is intended for educational and research purposes. Please ensure compliance with the licenses of all pretrained models and libraries used.
