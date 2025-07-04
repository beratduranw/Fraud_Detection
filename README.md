
# 🚀 Fraud Detection System

This project implements a robust fraud detection system using machine learning and real-time data processing. It integrates Apache Kafka for streaming, Apache Airflow for orchestration, MLflow for tracking, and Docker for deployment.

## 🌟 Features

- **Real-time Data Ingestion 📡**: Synthetic transactions with a 0.5%-ish fraud rate via Kafka.
- **Scheduled Training ⏰**: Airflow runs a daily pipeline for data processing and model training.
- **Feature Engineering 🧠**: Temporal, velocity, behavioral, and merchant-based features.
- **Class Imbalance ⚖️**: Handled using SMOTE.
- **Model Tuning 🔧**: XGBoost with randomized search for optimization.
- **Experiment Tracking 📊**: MLflow logs metrics and artifacts.
- **Real-time Inference ⚡**: Predicts fraud on incoming transactions.
- **Containerized Setup 🐳**: Managed with Docker Compose.

## 🏗️ Architecture

```
Adding the system architecture later on
```

## 📁 Project Structure

```
src/
├── .env                      # Environment variables
├── airflow/                  # Airflow setup
│   ├── Dockerfile
│   └── requirements.txt
├── config/                   # Configuration files
│   └── config.yaml
├── dags/                     # Airflow DAGs
│   ├── fraud_detection_training.py
│   └── fraud_detection_training_dag.py
├── docker-compose.yml        # Docker Compose setup
├── inference/                # Inference service
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── test.py
├── init-multiple-dbs.sh      # Database initialization script
├── mlflow/                   # MLflow setup
│   ├── Dockerfile
│   └── requirements.txt
├── models/                   # Trained models
│   ├── fraud_detection_model.pkl
│   └── preprocessor.pkl
├── plugins/                  # Airflow plugins
├── producer/                 # Transaction producer
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── test_consumer.log         # Test consumer logs
└── wait-for-it.sh            # Service wait script
```

## 🛠️ Prerequisites

- Docker 🐳
- Docker Compose 📦
- Git 🌿
- Confluent Kafka Cluster 🔗 (e.g., Confluent Cloud)

## 🚀 Installation

**Clone the Repo:**

```bash
git clone https://github.com/beratduranw/fraud-detection.git
cd fraud-detection
```

**Set Environment Variables:**

Create `.env` with your Kafka, Aws and MinIo settings.

**Start Containers:**

Use the Flower profile to include monitoring:

```bash
docker compose --profile flower up -d --build
```

## ⚙️ Configuration

- `config.yaml` 📝: Update Kafka and MLflow settings.
- `.env` 🔑: Add your credentials (Kafka, MinIO, etc.).

## 📊 Usage

After running `docker compose --profile flower up -d --build`, access:

- **Airflow UI 🌐**: http://localhost:8080
- **MLflow UI 📈**: http://localhost:5500
- **MinIO Console 🗄️**: http://localhost:9001
- **Flower UI 🌸**: http://localhost:5555

## 🏋️ Training

The DAG `fraud_detection_training` runs daily at 3 AM or can be triggered manually in the Airflow UI.

## 📦 Data Generation

The producer generates synthetic transactions with fraud patterns.

## 🤖 Inference

The inference service provides real-time predictions. Check its logs for output.

## 🔧 Inference Service Enhancements

The inference service is currently set up to consume transaction data from Kafka. However, it requires further development to fully function as intended. The planned upgrades include:

- **Feature Creation 🛠️**: Implement a mechanism to process streaming data in batches, creating the necessary features for prediction.
- **Model Prediction 🤖**: Integrate the trained model to make predictions on the processed feature batches.
- **Alerting Mechanism 🚨**: Develop a system to alert on predicted fraudulent transactions.
