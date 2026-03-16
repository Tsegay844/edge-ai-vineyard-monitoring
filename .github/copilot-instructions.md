# Copilot Instructions for Edge-AI Vineyard Monitoring

This repository contains an end-to-end edge-AI pipeline for vineyard monitoring, targeting ESP32 microcontrollers. Use the context below when providing code suggestions and answers.

## Project Overview

- **Goal**: Detect grape leaf diseases using lightweight deep-learning models deployed on ESP32 edge devices.
- **Hardware targets**: ESP32-P4, ESP32-S3.
- **Inference speed targets**: >18 FPS on ESP32-P4, >7 FPS on ESP32-S3.

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `leaf_detection/` | YOLOv11 / ESPDet-Pico grape-leaf detection (training, validation, export) |
| `dd_cnn/` | MobileNetV2 & ResNet disease-detection CNN models with INT8 quantisation |
| `esp-detection/` | Ultra-lightweight ESPDet-Pico framework (0.36 M params) |
| `Documentaion/` | LaTeX thesis documentation |
| `outputs_demo/` | Sample inference outputs |

## Key Technologies

- **Python** — PyTorch, Ultralytics YOLO, TensorFlow/Keras, OpenCV
- **C/C++** — ESP-IDF firmware for deployment on ESP32
- **Jupyter Notebooks** — Experimentation and demo pipelines
- **YAML** — Model and training configuration files

## Coding Conventions

- Follow **PEP 8** for Python code.
- Use **type hints** where practical.
- Prefer `pathlib.Path` over `os.path` for file operations.
- Keep model export scripts compatible with both ONNX and ESP-DL formats.
- Quantisation scripts should default to **INT8** for edge deployment.
- Document training hyper-parameters in the accompanying `.yaml` config file.

## Common Tasks

- Training a detection model: see `leaf_detection/train.py` and `esp-detection/` scripts.
- Running inference: see `leaf_detection/espdet_run.py`.
- Quantising a model for ESP32: see INT8 export utilities in `dd_cnn/` and `esp-detection/`.
- End-to-end demo: open `end_to_end_pipeline_demo_test.ipynb`.
