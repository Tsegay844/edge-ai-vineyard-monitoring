# Edge AI Computing for Grape Leaf Disease Detection In a LoRaWAN based IoT monitoring system

> **Master's Thesis Internship Work** | University of Pisa & INFN 
> **Hours:** 600 | 24 ECTS| **Score:** 30/30 (Maximum Score)

## Overview of the Work
This repository contains the code and full documentation for **EA-GDD** (Edge AI Grape Leaf Disease Detection), a low-power, vision-based IoT system for precision agriculture. Instead of transmitting heavy, raw images over limited LoRaWAN networks, this system runs a **dual-model deep learning pipeline directly on an ESP32-S3 microcontroller**. It analyzes grape leaves locally and transmits only a tiny, semantic text payload containing the final disease diagnosis.

## Key Technical Highlights
* **Hardware:** ESP32-S3 Microcontroller + OV3660 Camera Module.
* **Dual-Model Inference:** Fits two neural networks on a single MCU using INT8 Post-Training Quantization (ESP-DL).
* **High Accuracy:** 99.8% classification accuracy on cropped leaf patches.
* **Ultra-Low Power:** Estimated **10-month battery life** on a single 2400mAh Li-SOCl2 battery using a 6-hour deep-sleep duty cycle.
* **Data Reduction:** Sends bytes of data (aggregated diagnosis) over LoRaWAN instead of Megabytes of images.

## The 4-Stage Edge Pipeline
1. **Image Capture:** Acquires VGA-resolution (640x480) images of the vines.
2. **Localization (YOLO11n):** Uses a custom, INT8-quantized YOLO model (`ESPDet-Pico` - 478 KB) to detect and bound individual leaves (56.4% mAP@50).
3. **Classification (MobileNetV2):** Crops the detected leaves and passes them to an INT8 MobileNetV2 model (2.27 MB) to classify 4 states: *Black Rot, Esca, Healthy, and Leaf-Blight*.
4. **Aggregation & Transmission:** A weighted algorithm aggregates the multi-leaf inferences into a single diagnostic payload and transmits it via LoRaWAN.

##  Built With
* **Frameworks:** ESP-IDF, ESP-DL (for model deployment), PyTorch (for model training)
* **Models:** YOLO11n (Detection), MobileNetV2 (Classification)
* **IoT Protocols:** LoRaWAN



---
*Note: This repository was developed as part of an MSc Thesis in AI & Data Engineering at the University of Pisa in collaboration with the Italian National Institute for Nuclear Physics (INFN).*
