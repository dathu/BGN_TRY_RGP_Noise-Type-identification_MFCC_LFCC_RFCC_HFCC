# 🔊 Noise Type Identification CNN Classifier with cepstral features (MFCC, LFCC, RFCC, HFCC):  

This repository provides a **complete pipeline** for preparing, augmenting, training, and benchmarking noise type classification models using **Mel-Frequency Cepstral Coefficients (MFCCs), Linear Frequency Cepstral Coefficients (LFCC), Rectangular Frequency Cepstral Coefficients (RFCC), Human Factor Cepstral Coefficients (RFCC)** and **Convolutional Neural Networks (CNNs)**. It includes:  

- 📂 **Dataset structure & organization**  
- 🎧 **Speech data augmentation** (time-domain, spectrogram, and combination techniques)  
- ⚡ **Training with CNNs** (with/without SpecAugment)  
- 🧪 **Benchmarking on test data** with detailed reports  

---

## 📂 Dataset Folder Structure  

Datasets should be organized by **noise type** (e.g., Car, Exhibition, Station), with subfolders for **SNR levels**:  

```
dataset/
├── Car/
│ ├── 0dB/
│ │ ├── file1.wav
│ │ └── ...
│ ├── 5dB/
│ └── 10dB/
├── Exhibition/
│ ├── 0dB/
│ ├── 5dB/
│ └── 10dB/
└── Station/
├── 0dB/
├── 5dB/
└── 10dB/
```
After augmentation (explained below), new directories are created:  
```
dataset/
├── Car_augmented/
├── Exhibition_augmented/
└── Station_augmented/
```

---

## 🧪 Experimental Hardware and Software Environment Used for this Work

| **Component**              | **Specification / Version** |
|-----------------------------|------------------------------|
| 🖥️ **Operating System**        | ![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20LTS-E95420?logo=ubuntu&logoColor=white) |
| 💻 **Platform**                | Jupyter Hub |
| 🎮 **GPU**                     | ![NVIDIA](https://img.shields.io/badge/NVIDIA-A100%2080GB%20PCIe%20Gen-76B900?logo=nvidia&logoColor=white) |
| ⚙️ **CPU**                     | Intel Xeon Gold 6330 (28C, 205W, 2.0GHz) |
| 🧠 **RAM**                     | 32 × 16 = 512 GB |
| 🐍 **Python Version**          | ![Python](https://img.shields.io/badge/Python-3.10.12-3776AB?logo=python&logoColor=white) |
| 🔥 **Deep Learning Framework** | ![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C?logo=pytorch&logoColor=white) |
| 🎧 **Audio Library**           | ![TorchAudio](https://img.shields.io/badge/Torch%20Audio-Compatible-EE4C2C?logo=pytorch&logoColor=white) |
| 🧮 **Dataset Handling**        | ![NumPy](https://img.shields.io/badge/Numpy-1.x-013243?logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white) |

---

### 🧩 Summary
This setup provides a **high-performance environment** for deep learning and speech processing tasks, featuring:
- **Ubuntu 22.04 LTS** for stability and security  
- **NVIDIA A100 (80GB)** for large-scale GPU computations  
- **Intel Xeon Gold 6330** CPUs and **512GB RAM** for parallel data processing  
- **PyTorch + TorchAudio** for model training and audio feature extraction  
- **Numpy and Pandas** for efficient dataset handling and preprocessing  

---

✨ *Optimized for speech enhancement, noise classification, and real-time deep learning experiments.*

---

## 🎧 Speech Files Augmentation (Script: `0_speech_files_augmentation.py`)  

This script performs **data augmentation** to increase dataset diversity and robustness.  

### ✨ Augmentation Techniques
- **Time-Domain**: time stretching, pitch shifting, temporal shifting  
- **Spectrogram-Like**: band-stop filtering, clipping distortion  
- **Combination**: Gaussian noise overlay + pitch/time modifications  
- **Cropping & Padding**: normalize clip length to 2s  

### 🚀 How to Run
Update input/output dirs in the script:  
```
input_dir = "Station"          # Replace with Car, Exhibition, Station
output_dir = "Station_augmented"
```

Run:
```
0_python speech_files_augmentation.py
```
📊 Outputs

- time_*.wav → time-domain augmented
- spec_*.wav → spectrogram-like augmented
- crop_*.wav → cropped/padded
- combo_*.wav → combined augmentations

---

## ⚡ Training (Script: 1_Noise_type_identification_X.py) (Here X = MFCC, LFCC, RFCC, HFCC)

This script trains a CNN classifier using MFCC/LFCC/RFCCHFCC-based features extracted from noisy speech samples.

🚀 **Run Training**

**With SpecAugment:**
sudo /home/noise_type/noiseenv/bin/python3.10 1_Noise_type_identification_X.py --data_dir dataset --layers 6 --epochs 50 --specaugment

📊 Training Outputs

- metrics_*.csv → Training/validation metrics
- classification_view_*.csv → Predictions & probabilitie
- report_*.pdf → Multi-page training report (loss/accuracy, confusion matrix, ROC, calibration, PCA/t-SNE, misclassifications)
- noise_classifier_6layers_best.pth → Best saved model

---

## 🧪 Benchmarking (Script: 2_Noise_type_identification_X_benchmark.py)

This script evaluates a trained CNN model on a test dataset.

🚀 **Run Benchmarking:**

Make sure the trained model checkpoint exists (e.g., noise_classifier_6layers_best.pth):
sudo /home/noise_type/noiseenv/bin/python3.10 2_Noise_type_identification_X_benchmark.py

📊 **Benchmark Outputs**

- test_results.csv → File-level predictions
- classification_report.csv → Precision, Recall, F1 per class
- test_report.pdf → Multi-page evaluation report with:
  - Test accuracy summary
  - Per-class accuracy bar chart
  - Classification report table
  - Confusion matrix
  - ROC curves & AUC
  - Calibration curves

---

🔁 **End-to-End Workflow**

![Workflow Diagram](/Workflow.png)
---

📊 **Example Workflow:**

- Organize dataset (Car, Exhibition, Station with SNR subfolders)
- Augment speech files using speech_files_augmentation.py
- Train CNN using 1_Noise_type_identification_X.py
- Save model checkpoint (noise_classifier_6layers_best.pth)
- Benchmark model using 2_Noise_type_identification_X_benchmark.py
- Analyze results (CSV + PDF reports)

---
