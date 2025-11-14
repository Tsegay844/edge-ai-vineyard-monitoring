# Edge AI Computing for Grape Leaf Disease Detection in Vineyard Monitoring System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Master's Thesis Project**  
**Author:** Tsegay Teklay Gebrelibanos  
**Matriculation Number:** 683925  
**Program:** AIDE

## 📋 Overview

This project develops a long-range, low-power wireless sensor network (WSN) platform for vineyard environmental monitoring, enhanced with Edge AI capabilities for real-time grape leaf disease detection. The system combines LoRaWAN communication technology with on-device machine learning inference to provide comprehensive vineyard health monitoring.

## 🎯 Key Features

- **Environmental Monitoring**: Real-time measurement of temperature, pressure, soil moisture, and humidity  
- **Edge AI Disease Detection**: On-device CNN-based grape leaf disease classification using ESP32-S3  
- **LoRaWAN Communication**: Long-range, low-power data transmission in star network topology  
- **Microservices Architecture**: Containerized services orchestrated with Docker Compose  
- **Real-time Visualization**: Grafana dashboard for monitoring and analytics  
- **Cloud Integration**: MongoDB database for data storage and historical analysis

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ESP32-S3       │────▶│ Nucleo-WL55JC    │────▶│   LoRaWAN       │
│  (Edge AI)      │UART │ (End Node)       │     │   Gateway       │
│  - Image Capture│     │ - Environmental  │     └────────┬────────┘
│  - YOLO         │     │   Sensors        │              │
│  - CNN          │     │ - LoRa Tx        │              │
└─────────────────┘     └──────────────────┘              │
                                                           │
                    ┌──────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              Cloud Infrastructure                        │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐           │
│  │ The Things  │─▶│ Backend  │─▶│ MongoDB │           │
│  │ Stack (TTS) │  │ Services │  │         │           │
│  └─────────────┘  └─────┬────┘  └─────────┘           │
│                         │                               │
│                         ▼                               │
│                  ┌──────────┐                          │
│                  │ Grafana  │                          │
│                  │Dashboard │                          │
│                  └──────────┘                          │
└─────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
edge-ai-vineyard-monitoring/
├── docs/                           # Documentation
│   ├── thesis/                     # Thesis documentation
│   ├── api/                        # API documentation
│   └── hardware/                   # Hardware setup guides
├── firmware/                       # Embedded software
│   ├── esp32-s3/                   # Edge AI sensor code
│   │   ├── src/                    
│   │   │   ├── main.cpp
│   │   │   ├── camera.cpp
│   │   │   ├── yolo_inference.cpp
│   │   │   └── cnn_classifier.cpp
│   │   └── platformio.ini
│   └── nucleo-wl55jc/             # End node firmware
│       ├── Core/
│       ├── LoRaWAN/
│       └── Src/
├── models/                         # ML models
│   ├── training/                   # Model training scripts
│   │   ├── yolo_leaf_detection.py
│   │   └── cnn_disease_classification.py
│   ├── optimization/               # Model optimization for edge
│   │   ├── quantization.py
│   │   └── model_converter.py
│   └── pretrained/                 # Trained model weights
├── backend/                        # Backend services
│   ├── mqtt-client/               # MQTT data collector
│   ├── mongo-client/              # Database interface
│   ├── fastapi-server/            # REST API
│   └── docker-compose.yml
├── dashboard/                      # Grafana configuration
│   ├── dashboards/
│   └── datasources/
├── datasets/                       # Dataset management
│   ├── download_datasets.py
│   └── preprocessing/
├── tests/                          # Testing
│   ├── unit/
│   ├── integration/
│   └── power_analysis/
├── scripts/                        # Utility scripts
│   ├── deployment/
│   └── analysis/
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Hardware:**
  - ESP32-S3 (with camera module)
  - STM32 Nucleo-WL55JC
  - Environmental sensors (temperature, humidity, pressure, soil moisture)
  - LoRaWAN Gateway  
  
- **Software:**
  - Docker & Docker Compose
  - PlatformIO (for firmware development)
  - Python 3.8+
  - Node.js (optional, for some utilities)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tsegay844/edge-ai-vineyard-monitoring.git
   cd edge-ai-vineyard-monitoring
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download datasets:**
   ```bash
   python datasets/download_datasets.py
   ```

4. **Start backend services:**
   ```bash
   cd backend
   docker-compose up -d
   ```

5. **Flash firmware to devices:**
   ```bash
   # For ESP32-S3
   cd firmware/esp32-s3
   pio run --target upload
   
   # For Nucleo-WL55JC
   cd firmware/nucleo-wl55jc
   # Follow STM32 flashing instructions in docs/hardware/
   ```

## 🧠 Machine Learning Pipeline

### 1. Data Collection
- Grape leaf disease dataset from [Kaggle](https://www.kaggle.com/datasets/pushpalama/grape-disease)
- Augmented dataset from [HuggingFace](https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset)

### 2. Model Training
```bash
# Train YOLO for leaf detection
python models/training/yolo_leaf_detection.py

# Train CNN for disease classification
python models/training/cnn_disease_classification.py
```

### 3. Model Optimization
```bash
# Optimize models for ESP32-S3
python models/optimization/quantization.py
python models/optimization/model_converter.py
```

### 4. Deployment
Models are converted to TensorFlow Lite format and deployed to ESP32-S3 for edge inference.

## 📊 Data Flow

1. **Image Capture**: ESP32-S3 captures grape leaf images at specified intervals
2. **Leaf Detection**: YOLO model detects and crops grape leaves
3. **Disease Classification**: CNN model classifies leaf health status
4. **UART Communication**: Classification results sent to Nucleo-WL55JC
5. **LoRaWAN Transmission**: Combined environmental and AI data transmitted to gateway
6. **Cloud Processing**: TTS receives data, processes via MQTT, stores in MongoDB
7. **Visualization**: Grafana dashboard displays real-time and historical data

## 🔋 Power Consumption Analysis

The project includes comprehensive power consumption evaluation:
- Image capture phase
- Inference phase (YOLO + CNN)
- UART communication
- LoRaWAN transmission
- Sleep mode optimization

Results and analysis available in `tests/power_analysis/`

## 📈 Model Performance Metrics

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- Inference time
- Model size
- Confidence levels

## 🛠️ Technologies Used

- **Hardware**: ESP32-S3, STM32 Nucleo-WL55JC, LoRaWAN Gateway
- **Communication**: LoRaWAN, UART, MQTT
- **ML Frameworks**: TensorFlow/TensorFlow Lite, YOLOv5/v8
- **Backend**: FastAPI, Python
- **Database**: MongoDB
- **Visualization**: Grafana
- **Containerization**: Docker, Docker Compose
- **LoRaWAN Server**: The Things Stack (TTS)

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:
- [System Architecture](docs/architecture.md)
- [Hardware Setup](docs/hardware/)
- [API Reference](docs/api/)
- [Model Training Guide](docs/ml-pipeline.md)
- [Power Optimization](docs/power-optimization.md)

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run power consumption tests
python tests/power_analysis/measure_consumption.py
```

## 🤝 Contributing

This is a thesis project, but suggestions and feedback are welcome! Please feel free to:
1. Open an issue for bugs or suggestions
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Tsegay Teklay Gebrelibanos**  
AIDE Program - Matriculation Number: 683925

## 🙏 Acknowledgments

- Datasets: Kaggle and HuggingFace grape leaf disease datasets
- LoRaWAN: The Things Network community
- Edge AI: ESP32 community and TensorFlow Lite team

## 📚 Citations

If you use this work in your research, please cite:

```bibtex
@mastersthesis{gebrelibanos2025edge,
  title={Edge AI Computing for Grape Leaf Disease Detection in a Vineyard Monitoring System},
  author={Gebrelibanos, Tsegay Teklay},
  year={2025},
  school={[Your University]},
  type={Master's Thesis}
}
```

## 🔗 Related Links

- [Grape Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/pushpalama/grape-disease)
- [Augmented Dataset (HuggingFace)](https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset)
- [The Things Stack Documentation](https://www.thethingsindustries.com/docs/)
- [ESP32-S3 Documentation](https://www.espressif.com/en/products/socs/esp32-s3)

## 📅 Project Timeline

- **Phase 1**: System design and architecture (Completed)
- **Phase 2**: Environmental monitoring implementation (Completed)
- **Phase 3**: ML model development and training (In Progress)
- **Phase 4**: Edge AI integration (Upcoming)
- **Phase 5**: Testing and evaluation (Upcoming)
- **Phase 6**: Thesis writing and defense (Upcoming)

---

**Status**: 🚧 Work in Progress - Thesis Project 2025
