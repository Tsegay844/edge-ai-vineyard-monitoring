# Edge AI Implementation Plan

**Implementer:** Tsegay Teklay Gebrelibanos  
**Matriculation Number:** 683925  
**Start Date:** 2025-11-14  
**Focus:** Edge AI Computing for Grape Leaf Disease Detection

## Overview

This document outlines the implementation plan for integrating Edge AI capabilities into the existing vineyard monitoring system. The focus is on implementing real-time grape leaf disease detection using ESP32-S3 with on-device machine learning inference.

## Project Scope

### In Scope (Your Implementation)
1. **ML Model Development**
   - YOLO model for grape leaf detection
   - CNN model for disease classification
   - Model training and validation

2. **Model Optimization**
   - Quantization for edge deployment
   - Conversion to TensorFlow Lite
   - Model size and inference time optimization

3. **ESP32-S3 Edge AI Sensor**
   - Camera integration and image capture
   - On-device YOLO inference
   - On-device CNN classification
   - UART communication with Nucleo-WL55JC
   - Power optimization and sleep modes

4. **Integration**
   - UART protocol design for sensor communication
   - Data packet format for disease detection results
   - Integration with existing LoRaWAN data stream

5. **Evaluation**
   - Power consumption analysis
   - Model accuracy and performance metrics
   - Inference time measurement
   - End-to-end system testing

### Out of Scope (Handled by Colleague)
- LoRaWAN network infrastructure
- Environmental sensor integration
- Backend services (MQTT, MongoDB, FastAPI)
- Grafana dashboard
- The Things Stack server configuration

## Implementation Phases - STEP BY STEP

### ✅ PHASE 0: Repository Setup (COMPLETED)
- [x] Repository created
- [x] README.md added
- [x] .gitignore configured
- [x] requirements.txt created
- [x] Implementation plan documented

### 📍 PHASE 1: Dataset Preparation (CURRENT - Week 1-2)

**Objective:** Download, explore, and prepare grape leaf disease datasets

**Tasks:**
1. [ ] Download Kaggle dataset (pushpalama/grape-disease)
2. [ ] Download HuggingFace dataset (adamkatchee/grape-leaf-disease-augmented-dataset)
3. [ ] Create data exploration notebook
4. [ ] Analyze disease class distribution
5. [ ] Identify data quality issues
6. [ ] Document dataset statistics

**Deliverables:**
- `datasets/download_datasets.py` ✅
- `datasets/explore_data.ipynb`
- `datasets/DATASET_INFO.md`
- Data analysis report

**Next Step:** Run `python datasets/download_datasets.py`

---

### PHASE 2: Data Preprocessing (Week 2-3)

**Objective:** Prepare data for model training

**Tasks:**
1. [ ] Create train/validation/test splits (70/15/15)
2. [ ] Implement data augmentation pipeline
3. [ ] Create data loaders for PyTorch/TensorFlow
4. [ ] Normalize and resize images
5. [ ] Handle class imbalance

**Deliverables:**
- `datasets/preprocessing/data_loader.py`
- `datasets/preprocessing/augmentation.py`
- `datasets/preprocessing/split_dataset.py`

---

### PHASE 3: YOLO Model Training (Week 3-4)

**Objective:** Train YOLO model for grape leaf detection

**Tasks:**
1. [ ] Annotate images for leaf detection (if needed)
2. [ ] Select YOLO architecture (YOLOv8 nano/small)
3. [ ] Configure training parameters
4. [ ] Train model on dataset
5. [ ] Validate and test model
6. [ ] Calculate mAP and other metrics

**Deliverables:**
- `models/training/yolo_leaf_detection.py`
- `models/training/yolo_config.yaml`
- Trained YOLO weights
- Training logs and visualizations

**Target Metrics:**
- mAP > 85%
- Inference time < 300ms (on PC)

---

### PHASE 4: CNN Disease Classification (Week 5-6)

**Objective:** Train CNN model for disease classification

**Tasks:**
1. [ ] Design CNN architecture (MobileNetV2/EfficientNet)
2. [ ] Implement transfer learning
3. [ ] Train model on disease classes
4. [ ] Validate and test model
5. [ ] Generate confusion matrix
6. [ ] Analyze per-class performance

**Deliverables:**
- `models/training/cnn_disease_classification.py`
- `models/training/cnn_architecture.py`
- Trained CNN weights
- Evaluation metrics and reports

**Target Metrics:**
- Accuracy > 90%
- Balanced precision/recall across classes

---

### PHASE 5: Model Optimization for Edge (Week 7-8)

**Objective:** Optimize models for ESP32-S3 deployment

**Tasks:**
1. [ ] Implement post-training quantization (INT8)
2. [ ] Convert models to TensorFlow Lite format
3. [ ] Test quantized model accuracy
4. [ ] Benchmark inference speed
5. [ ] Optimize model size
6. [ ] Document accuracy vs size trade-offs

**Deliverables:**
- `models/optimization/quantization.py`
- `models/optimization/model_converter.py`
- `models/optimization/benchmark.py`
- Optimized .tflite models
- Optimization report

**Target Metrics:**
- Model size < 1MB each
- Accuracy drop < 5%
- Inference time < 500ms on ESP32-S3

---

### PHASE 6: ESP32-S3 Setup (Week 9)

**Objective:** Set up ESP32-S3 development environment

**Tasks:**
1. [ ] Install PlatformIO
2. [ ] Set up ESP32-S3 board configuration
3. [ ] Test camera module (OV2640)
4. [ ] Implement basic image capture
5. [ ] Test TensorFlow Lite Micro
6. [ ] Verify UART communication

**Deliverables:**
- `firmware/esp32-s3/platformio.ini`
- `firmware/esp32-s3/test/camera_test.cpp`
- `firmware/esp32-s3/test/uart_test.cpp`
- Hardware setup documentation

---

### PHASE 7: Edge Inference Implementation (Week 10-11)

**Objective:** Implement on-device ML inference

**Tasks:**
1. [ ] Integrate TensorFlow Lite Micro
2. [ ] Load YOLO model on ESP32-S3
3. [ ] Load CNN model on ESP32-S3
4. [ ] Implement image preprocessing
5. [ ] Implement YOLO inference
6. [ ] Implement CNN inference
7. [ ] Test end-to-end pipeline

**Deliverables:**
- `firmware/esp32-s3/src/main.cpp`
- `firmware/esp32-s3/src/camera.cpp`
- `firmware/esp32-s3/src/yolo_inference.cpp`
- `firmware/esp32-s3/src/cnn_classifier.cpp`
- `firmware/esp32-s3/src/image_processing.cpp`

---

### PHASE 8: UART Communication (Week 12)

**Objective:** Implement UART protocol for data transmission

**Tasks:**
1. [ ] Design UART protocol specification
2. [ ] Define data packet format
3. [ ] Implement UART transmitter (ESP32-S3)
4. [ ] Coordinate with colleague for receiver (Nucleo-WL55JC)
5. [ ] Test communication reliability
6. [ ] Add error checking (CRC)

**Deliverables:**
- `firmware/esp32-s3/src/uart_comm.cpp`
- `docs/uart_protocol.md`
- UART test results
- Protocol specification document

**Data Packet Format:**
```
[HEADER][DISEASE_CLASS][CONFIDENCE][CRC][FOOTER]
Example: "LEAF|Black_Rot|0.92|A5|END"
```

---

### PHASE 9: Power Optimization (Week 13)

**Objective:** Optimize power consumption for battery operation

**Tasks:**
1. [ ] Implement deep sleep mode
2. [ ] Configure wake-up intervals
3. [ ] Optimize inference frequency
4. [ ] Measure power consumption in each phase
5. [ ] Calculate battery lifetime
6. [ ] Document power optimization strategies

**Deliverables:**
- `firmware/esp32-s3/src/power_mgmt.cpp`
- `tests/power_analysis/measure_consumption.py`
- `tests/power_analysis/power_report.md`
- Power consumption graphs
- Battery lifetime calculations

**Target Metrics:**
- Active current < 500mA
- Sleep current < 10mA
- Battery life > 6 months (with hourly sampling)

---

### PHASE 10: Integration Testing (Week 14)

**Objective:** Test complete system integration

**Tasks:**
1. [ ] Integration with Nucleo-WL55JC
2. [ ] End-to-end data flow testing
3. [ ] Test with real vineyard images
4. [ ] Verify LoRaWAN transmission
5. [ ] Check MongoDB data storage
6. [ ] Validate Grafana visualization

**Deliverables:**
- `tests/integration/test_edge_ai.py`
- Integration test report
- System performance analysis
- Bug fixes and improvements

---

### PHASE 11: Field Testing (Week 15)

**Objective:** Test system in real-world conditions

**Tasks:**
1. [ ] Deploy system in test environment
2. [ ] Collect real-world data
3. [ ] Monitor system reliability
4. [ ] Analyze false positives/negatives
5. [ ] Fine-tune models if needed
6. [ ] Document field test results

**Deliverables:**
- Field test report
- Real-world performance metrics
- Reliability analysis
- Lessons learned

---

### PHASE 12: Documentation & Thesis (Week 16-18)

**Objective:** Complete technical documentation and thesis

**Tasks:**
1. [ ] Write technical documentation
2. [ ] Create user guide
3. [ ] Document API
4. [ ] Write thesis chapters
5. [ ] Prepare results and analysis
6. [ ] Create presentation
7. [ ] Prepare for defense

**Deliverables:**
- Complete technical documentation
- User guide
- Thesis draft
- Presentation slides
- Demo video

---

## Weekly Progress Checklist

### Week 1 (2025-11-14 to 2025-11-21) - CURRENT WEEK
- [x] Set up repository
- [x] Create implementation plan
- [ ] Set up local development environment
- [ ] Download datasets
- [ ] Begin data exploration
- [ ] Create exploration notebook

### Week 2 (2025-11-22 to 2025-11-29)
- [ ] Complete data analysis
- [ ] Create preprocessing pipeline
- [ ] Split datasets
- [ ] Implement augmentation

---

## Success Criteria

Each phase must meet these criteria before moving to the next:

1. ✅ **Code Quality**
   - Clean, documented code
   - Follows Python/C++ best practices
   - Version controlled in Git

2. ✅ **Testing**
   - Unit tests written
   - Integration tests passing
   - Performance benchmarks met

3. ✅ **Documentation**
   - README updated
   - Code comments added
   - Technical decisions documented

4. ✅ **Deliverables**
   - All files committed to GitHub
   - Results analyzed and documented
   - Metrics calculated and recorded

---

## Next Immediate Steps (RIGHT NOW)

### Step 1: Clone Repository Locally
```bash
git clone https://github.com/Tsegay844/edge-ai-vineyard-monitoring.git
cd edge-ai-vineyard-monitoring
```

### Step 2: Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Set Up Kaggle API
1. Go to https://www.kaggle.com/settings/account
2. Create new API token
3. Save kaggle.json to ~/.kaggle/kaggle.json

### Step 4: Download Datasets
```bash
python datasets/download_datasets.py
```

### Step 5: Start Data Exploration
```bash
jupyter notebook datasets/explore_data.ipynb
```

---

## Communication with Colleague

**Coordinate on:**
1. UART protocol format
2. LoRaWAN packet structure
3. MongoDB schema for disease data
4. Grafana dashboard updates

**Meeting Schedule:** Weekly sync-up (suggested: Fridays)

---

**Last Updated:** 2025-11-14 15:33:52 UTC  
**Status:** Phase 1 - Dataset Preparation  
**Next Milestone:** Complete data exploration by 2025-11-21
