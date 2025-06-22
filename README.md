---
title: Which Simpsons Character Are You
emoji: 🍩
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: mit
short_description: Image classifier used as an entertainment app
---

# 🍩 Which Simpsons Character Are You?

**Live Demo**: [Hugging Face Space →](https://huggingface.co/spaces/Vukodlok/Which_Simpsons_Character_Are_You)

This project is an image classification app that predicts which Simpsons character you most resemble based on a photo. It was built as a fun and interactive showcase of computer vision techniques and transformer-based models, suitable for demonstrating end-to-end machine learning engineering skills.

## Overview

- **Model Architecture**: Vision Transformer (ViT)
- **Pretrained Base**: [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)
- **Task**: Multi-class image classification
- **Classes**: 42 characters from *The Simpsons*
- **Interface**: Built with [Gradio](https://www.gradio.app/), deployed via Hugging Face Spaces

Upload an image (or take a selfie), and the app returns the top 3 predicted characters with confidence scores.

## Features

- Fine-tuned transformer for visual recognition
- Fast and interactive UI using Gradio
- Responsive design for both desktop and mobile
- Custom styling to match The Simpsons’ theme
- Easily extendable or adaptable to new character datasets

## How It Works

1. The image is preprocessed to match the ViT input format.
2. A fine-tuned ViT model performs inference.
3. The top 3 characters with the highest probabilities are displayed.
4. Optional webcam support (best used on desktop browsers).

> Note: On mobile, it's recommended to take a photo using your camera app and upload it directly for best experience.

## Dataset Credit

This model was trained on the [Simpsons Character Data](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset) created by **AlexAttia** on Kaggle. It contains thousands of labeled images of Simpsons characters and is licensed for educational use.

## Files

- `app.py` – Main Gradio interface and prediction logic
- `simpsons_model.pt` – Fine-tuned PyTorch model (loaded at runtime)
- `README.md` – Project documentation and metadata
- `requirements.txt` – Dependency list (managed by Hugging Face automatically)

## Purpose

This project was developed as a portfolio piece to demonstrate:

- Transfer learning and fine-tuning of transformer models
- Serving ML models in a production-ready interface
- Building engaging ML-powered applications
- Gradio UI customization and deployment workflows

## License

This project is licensed under the MIT License. It is intended for educational and entertainment purposes.
