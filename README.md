# Cognitive IDS — Anomaly Detection System

A comprehensive network intrusion detection system (NIDS) that leverages an ensemble of Deep Learning architectures to detect anomalous network traffic in real-time. 

This project trains on standard datasets like **CICIDS2017** and **NSL-KDD**, and uses an advanced meta-classifier ensemble consisting of BiLSTM, CNN, Transformer, and Variational Autoencoder (VAE) models.

---

## 🌟 Key Features

- **Multi-Model Ensemble Architecture**: Combines probabilities from BiLSTM, CNN, Transformer, and VAE models using an XGBoost meta-classifier for robust anomaly detection.
- **Real-Time Dashboard**: A Flask-based web dashboard to visualize network traffic and anomaly alerts in real-time.
- **Traffic Simulation**: Includes `send_traffic.py` to stream preprocessed test data to the dashboard and simulate a live network environment.
- **Local Training Pipeline**: A fully resumable local training pipeline with support for SMOTE oversampling and checkpointing.

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SamyakBhongade/Anomaly_Detection_CICIDS.git
   cd Anomaly_Detection_CICIDS
   ```

2. **Install dependencies:**
   Requires Python 3.8+.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Flask and requests are also required for the dashboard `pip install Flask requests`)*

3. **⚠️ IMPORTANT: Download Inference Models**
   Because trained `.keras` neural networks and the `.pkl` scalers are too large for GitHub, they are excluded via `.gitignore`. **If you cloned this repo, you will need the `inference_models` directory to make predictions.**
   - **Option A:** Download the `inference_models.zip` from the [Releases page](https://github.com/SamyakBhongade/Anomaly_Detection_CICIDS/releases) (if provided by the author) and extract it into the project root.
   - **Option B:** Train the models yourself locally (see the Training section below).

---

## 🚀 Usage

### 1. Real-time Dashboard
Start the real-time monitoring dashboard server:
```bash
python dashboard_app/app.py
```
*The dashboard will be available at `http://localhost:5000`.*

### 2. Traffic Simulator
In a separate terminal, run the traffic simulator to send traffic to the dashboard:
```bash
python send_traffic.py
```
This will send batches of network flows (mixed benign and attacks) to the dashboard for live detection.

### 3. Command Line Inference
If you have a CSV of raw network flows (70 features, no labels), you can predict anomalies directly:
```bash
python predict.py --input data/sample_network_traffic.csv --output predictions.csv
```

---

## 🧠 Training Your Own Models

If you wish to train the ensemble from scratch on the CICIDS2017 or NSL-KDD datasets, please refer to the detailed **[Local Training Guide](LOCAL_TRAINING_GUIDE.md)**.

**Quick Start (CICIDS2017):**
```bash
python train_local.py --dataset cicids
```
*Training supports auto-resume. You can stop it at any time with `Ctrl+C` and restart it later!*

---

## 📂 Project Structure

```text
cognitive_ids/
├── dashboard_app/         # Flask server & web dashboard UI
├── src/                   # Model architectures & preprocessing pipelines
│   ├── bilstm_model.py
│   ├── cnn_model.py
│   ├── transformer_model.py
│   ├── vae_model.py
│   ├── ensemble.py
│   └── preprocess.py
├── inference_models/      # Saved trained models & scalers (Not tracked in Git)
├── train_local.py         # Script to train models locally
├── predict.py             # Inference API and CLI script
├── send_traffic.py        # Script to simulate live network traffic
├── LOCAL_TRAINING_GUIDE.md# Detailed training instructions
└── requirements.txt       # Python dependencies
```
