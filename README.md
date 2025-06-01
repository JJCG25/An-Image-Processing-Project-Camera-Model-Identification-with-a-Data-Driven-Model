<p align="center">
  <h1 align="center">Camera Model Identification with a Data-Driven Model</h1>
  <p align="center">
    <img src="./data/results.png" alt="Panorama Image" width="1600">
  <p align="center">
    <a href="https://github.com/JJCG25" rel="external nofollow noopener" target="_blank"><strong>Juan Calderón</strong></a>
    ·
    <a href="https://github.com/Danita21" rel="external nofollow noopener" target="_blank"><strong>Dana Villamizar</strong></a>
    ·
    <a href="https://github.com/Sneider-exe" rel="external nofollow noopener" target="_blank"><strong>Sneider Sánchez</strong></a>
  </p>
<p align="center">
    Digital Image Processing project at UIS 2025-1

- Visit the  [report](report.pdf) for more information about the project.
- Visit the  [slides](slides.pdf) for the presentation.

## Overview

This repository contains the source code and trained models for the project titled **"Camera Model Identification with a Data-Driven Model"**. The goal of the project is to classify the smartphone brand or model that captured a digital image using convolutional neural networks.

The project implements two main strategies:
1. Replication of a CNN architecture proposed by [Baroffio *et al.*](https://www.researchgate.net/publication/301841474_Camera_identification_with_deep_convolutional_networks) for identifying traditional camera models.
2. Fine-tuning of a pre-trained EfficientNetV2-M model on a smartphone image dataset (FloreView) for brand-level classification.

The model achieves a validation accuracy of **82%** on the FloreView dataset and is available through a simple web [interface](https://www.researchgate.net/publication/301841474_Camera_identification_with_deep_convolutional_networks).

A detailed explanation of the methodology and results can be found in the [final paper (PDF)](Camera Model Identification with a Data-Driven Model.pdf).

---

## Datasets

- **Training 1:** [Dresden Image Database](https://www.kaggle.com/datasets/micscodes/dresden-image-database)
- **Training 2:** [FloreView Dataset](https://lesc.dinfo.unifi.it/FloreView/) – smartphone images acquired under controlled conditions.

---

## Model Options

The project includes two model implementations:

- `CameraConvNet`: Custom CNN from Baroffio *et al.*, trained from scratch and later fine-tuned on smartphone images.
- `EfficientNetV2_M`: Pre-trained on ImageNet and fully fine-tuned on the FloreView dataset for mobile camera classification.

🔗 **Download Trained Models**  
You can download the `.pth` files for both models from the following Google Drive folder:  
[Google Drive – Trained Models](https://drive.google.com/drive/folders/1yU4jWUso9WsH4zSfsF7ptDyKon2optAi?usp=sharing)

---
## Installation

To set up the environment and install dependencies:

```bash
git clone https://github.com/JJCG25/An-Image-Processing-Project-Camera-Model-Identification-with-a-Data-Driven-Model
cd An-Image-Processing-Project-Camera-Model-Identification-with-a-Data-Driven-Model
pip install -r requirements.txt
