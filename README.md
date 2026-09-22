---
title: MLTrainer
emoji: 🔬
colorFrom: cyan
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Machine Vision Model Training Studio in Python
---

# MLTrainer 🔬
### Machine Vision Model Training Studio in Python

**MLTrainer** is a comprehensive, production-grade machine vision studio designed for training, fine-tuning, evaluating, and exporting computer vision models in Python.

Built with a high-performance **FastAPI + PyTorch backend** and a modern **React 19 + TypeScript + Vite frontend**, MLTrainer provides an intuitive, glassmorphic dark-mode interface for end-to-end computer vision workflows.

---

## 🚀 Key Features & Supported Tasks

| Task | Description | Default Backbones | Metrics |
| :--- | :--- | :--- | :--- |
| **🎯 Image Categorization** | Multi-class & single-label image classification | ResNet-18, MobileNet-V3, EfficientNet-B0, Custom CNN | Accuracy, Precision, Recall, F1, Confusion Matrix |
| **📦 Object Detection** | Bounding box localization and multi-object labeling | TinyYOLO Grid Head, ResNet-18 FPN Head | mAP@0.5, mAP@0.5:0.95, Precision, Recall |
| **🔍 Anomaly Detection** | Visual defect inspection & reconstruction residuals | Conv Autoencoder, PatchCore Feature Extractor | AUROC, Pixel AUROC, F1-max, Decision Threshold |
| **🧩 Segmentation** | Pixel-wise semantic & instance mask segmentation | U-Net Encoder-Decoder, DeepLabV3+ | Mean IoU (mIoU), Dice Score, Pixel Accuracy |

---

## 📦 Core Studio Modules

### 1. Image Gallery & Annotation Studio (`Module 1`)
- **Visual Dataset Grid**: Filter by dataset partition (*Train*, *Validation*, *Test*), search by filename, and filter by class tags.
- **Interactive Lightbox**: Full inspection view with toggles for:
  - Bounding box overlays + confidence tags (*Object Detection*)
  - Anomaly heatmaps (*Anomaly Inspection*)
  - Multi-polygon segmentation masks (*Segmentation*)
- **In-place Annotation Editor**: Reassign classes, adjust defect ground truth, and modify partitions.
- **Data Uploader**: Drag-and-drop single images (PNG, JPEG, WEBP) or bulk ZIP archives with automatic extraction.

### 2. Dataset Partitioning & Stratification (`Module 2`)
- **Configurable Split Ratios**: Interactive sliders for Train, Validation, and Test subsets with quick presets (70/15/15, 80/10/10, 60/20/20).
- **Stratified Sampling**: Automatically preserves proportional class representations across all splits to prevent dataset skew.
- **Reproducible Seed Control**: Set random seeds for deterministic, repeatable experiments.
- **Class Distribution Table**: Visual breakdown and percentage indicators for each class across train, validation, and test sets.

### 3. Training Studio & Live Telemetry (`Module 3`)
- **Hyperparameter Controls**:
  - Model architecture / backbone selection
  - Pretrained ImageNet transfer learning toggle
  - Resolution (224x224, 384x384, 512x512)
  - Batch size (4, 8, 16, 32)
  - Optimizer (AdamW, Adam, SGD with momentum)
  - Learning rate scheduler (Cosine Annealing, StepLR)
  - Data augmentations (Random Flip, Rotation, Color Jitter, MixUp/CutMix)
- **Live WebSocket Telemetry**:
  - Real-time animated Loss convergence curves (Train vs Validation Loss)
  - Metric progression curves (Accuracy, mAP@50, AUROC, mIoU)
  - Progress bars, elapsed time, step counters, and ETA
  - Hardware status badge (CPU / CUDA GPU detection)
- **Session Controls**: Start, Pause, Resume, Abort/Stop, and automatic checkpoint persistence.

### 4. Evaluation & Diagnostic Studio (`Module 4`)
- **Primary & Secondary Metrics**: High-contrast scorecards for primary accuracy/mAP/IoU/AUROC, precision, recall, and test sample counts.
- **Interactive Confusion Matrix**:
  - Visual matrix highlighting true positives (green/cyan) and misclassifications (rose/amber).
  - Click any matrix cell to filter predictions and inspect the exact images that were misclassified!
- **ROC Curves & Anomaly Threshold Tuner**: Interactive slider to adjust defect sensitivity threshold and view real-time impact on false alarms.
- **Visual Prediction Inspector**: Side-by-side comparison of Ground Truth annotations vs Model Predicted outputs.

### 5. Model Export & Deployment Hub (`Module 5`)
- **Serialized Formats**:
  - `ONNX (.onnx)`: Cross-platform deployment for TensorRT, OpenVINO, and ONNX Runtime.
  - `TorchScript (.pt)`: Compiled JIT graph for C++ and Python production servers.
  - `PyTorch (.pth)`: State dict weights for continuous fine-tuning.
- **Quantization**: FP32 (Standard), FP16 (Half precision for GPUs), and INT8 (Dynamic quantization for edge/CPU).
- **Inference Code Generator**: Auto-generated, standalone runnable `inference.py` script tailored to the model task and class mappings.
- **REST Microservice**: Ready-to-deploy FastAPI `serve.py` snippet.
- **One-Click Download**: Bundles model weights, `classes.json`, `config.json`, `inference.py`, `serve.py`, and `README.md` into a single `.zip` archive.

---

## ⚡ Quick Start

### Windows (One-Click)
Simply run the bundled `start.bat` script:
```cmd
start.bat
```
This will automatically:
1. Validate Python 3.10+ and Node.js.
2. Create and activate the Python virtual environment in `backend\.venv`.
3. Install backend packages from `backend\requirements.txt`.
4. Install frontend npm dependencies in `frontend\node_modules`.
5. Launch the FastAPI server on `http://127.0.0.1:8000`.
6. Launch Vite on `http://localhost:3000` and open your browser.

---

### Manual Launch

#### 1. Backend Server
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend Server
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## 📁 Project Structure

```
MLTrainer/
├── start.bat                         # One-click turnkey launcher script
├── README.md                         # Project documentation
├── backend/
│   ├── config.py                     # Configuration paths & constants
│   ├── main.py                       # FastAPI REST API & WebSocket endpoints
│   ├── requirements.txt              # Python dependencies
│   ├── core/
│   │   ├── dataset_manager.py        # Dataset handling & pre-loaded demo generators
│   │   ├── split_manager.py          # Stratified splitting & partition metrics
│   │   ├── trainer.py                # Async training engine & WebSocket telemetry
│   │   ├── evaluator.py              # Confusion matrix, ROC, PR, and prediction inspector
│   │   ├── exporter.py               # ONNX/TorchScript/ZIP packaging & code generation
│   │   └── models/
│   │       ├── classifier.py         # ResNet, MobileNet, EfficientNet, Custom CNN
│   │       ├── detector.py           # TinyYOLO grid detector
│   │       ├── segmenter.py          # U-Net encoder-decoder
│   │       └── anomaly.py            # Conv Autoencoder & PatchCore
│   └── demo_data/                    # Storage for demo & custom images
└── frontend/
    ├── package.json                  # React 19, Lucide, Vite
    ├── vite.config.ts                # Vite proxy configuration
    ├── tsconfig.json                 # TypeScript compiler options
    ├── src/
    │   ├── App.tsx                   # Main layout & WebSocket manager
    │   ├── index.css                 # Dark machine vision design system
    │   ├── types/index.ts            # TypeScript data interfaces
    │   ├── services/api.ts           # Typed API client
    │   └── components/
    │       ├── Navbar.tsx            # Header, task selector, telemetry pill
    │       ├── Gallery/
    │       │   ├── ImageGallery.tsx  # Grid view, filters, search
    │       │   ├── ImageModal.tsx    # Lightbox, overlays & annotation editor
    │       │   └── DatasetUploader.tsx # Drag-and-drop & ZIP uploader
    │       ├── Splitting/
    │       │   └── SplitEditor.tsx   # Partition sliders & stratification
    │       ├── Training/
    │       │   └── TrainingStudio.tsx # Hyperparameters, loss curves, controls
    │       ├── Evaluation/
    │       │   └── EvaluationDashboard.tsx # Confusion matrix & prediction inspector
    │       └── Export/
    │           └── ExportHub.tsx     # ONNX export, code generator, ZIP download
```

---

## 🛡️ Pre-Loaded Demo Datasets

MLTrainer includes 4 pre-generated machine vision datasets ready immediately upon first launch:
1. **Categorization**: 28 industrial mechanical components categorized into *Normal*, *Scratch*, *Dent*, and *Void*.
2. **Object Detection**: 20 PCB circuit boards with bounding boxes for *Resistor*, *Capacitor*, *IC_Chip*, and *Diode*.
3. **Anomaly Inspection**: 24 metallic surface textures with continuous anomaly scores and visual defect heatmaps.
4. **Segmentation**: 20 micro-cellular and part specimens with multi-class polygon masks (*Defect_Core*, *Surface_Flaw*).
