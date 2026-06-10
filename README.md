# Generative AI in Computer Vision

## Course Overview

This advanced microcredit introduces students to the core ideas behind generative AI for computer vision. The course is delivered through two executable Jupyter notebooks: one notebook explains the theory and builds intuition through small implementations, and the second notebook presents a DreamBooth case study for fine-tuning a diffusion model with a small image dataset.

Students first review the image-processing operations that make generative vision models possible, including upsampling, transposed convolution, autoencoders, variational autoencoders, and U-Net-style architectures. The course then moves into diffusion models, covering the forward noising process, the learned reverse denoising process, DDPM-style sampling, and the practical system components used in modern text-to-image pipelines.

## Learning Objectives

After completing this microcredit, students should be able to:

- Explain how upsampling, interpolation, max unpooling, and transposed convolution change image resolution.
- Describe encoder-decoder architectures and explain why autoencoders are useful precursors to generative models.
- Distinguish between autoencoders, variational autoencoders, GANs, and diffusion models at a conceptual level.
- Explain the forward and reverse processes used in denoising diffusion probabilistic models.
- Connect the theory of diffusion models to practical components such as U-Net denoisers, schedulers, prompts, and fine-tuning.
- Run and interpret notebook experiments for image generation, denoising, and DreamBooth-based personalization.

## Course Structure

1. **Theory and Implementation Notebook**
   - Upsampling methods for image processing
   - Autoencoders and variational autoencoders
   - U-Net architecture for dense prediction and denoising
   - Diffusion model foundations
   - Forward and reverse diffusion examples
   - Simple implementation exercises

2. **DreamBooth Case Study Notebook**
   - Small-dataset personalization
   - Hugging Face Diffusers workflow
   - Dataset loading from the Hugging Face Hub
   - Fine-tuning considerations for GPU environments
   - Comparison of model behavior before and after fine-tuning

## Prerequisites

This is an advanced microcredit. Students are expected to have prior knowledge of:

- Python programming
- NumPy and Matplotlib basics
- Machine learning fundamentals
- Convolutional neural networks
- Jupyter Notebook usage

Recommended prior microcredits:

- Single Layer Perceptron (SLP)
- Multi Layer Perceptron (MLP)
- CNN Basics

## Setup


## Repository

Clone the course repository:

```bash
git clone https://gitlab.gwdg.de/ki4all/ohm11_genai.git

```

Open the notebooks from the `notebooks` directory. The main concept notebook expects a local `requirements.txt` file, which is provided in the same directory:

```bash
cd notebooks
pip install -r requirements.txt
```

The DreamBooth case study requires a CUDA-capable GPU for practical training. The notebook notes a minimum of approximately 14 GB GPU VRAM for the simplified workflow.


## Workload

**Extent / Arbeitsumfang:** 1 ECTS

## Responsibility

**Responsible / Verantwortlichkeit:** Ostfalia Hochschule fuer angewandte Wissenschaften, Informatik, Srinivas Kachavarapu (MSc)
