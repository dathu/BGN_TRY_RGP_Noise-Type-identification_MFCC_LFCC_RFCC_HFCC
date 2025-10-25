'''
sudo /home/raghu/raghu/bgnenv/bin/python3.10 2a_Noise_type_identification_24092025_RFCC_model_load_benchmark.py \
  --data_dir noisezus_30speakers \
  --model_path noise_classifier_RFCC_6layers_best.pth \
  --layers 6 \
  --batch_size 16
'''
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import librosa

# =====================================================
# Noise classes
# =====================================================
noise_types = ['Car', 'Exhibition', 'Station']

# =====================================================
# Utility functions
# =====================================================
def pad_truncate_waveform(waveform, target_length, sample_rate):
    length = waveform.shape[-1]
    if length > target_length:
        waveform = waveform[..., :target_length]
    elif length < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - length))
    return waveform

def load_data(data_dir='.'):
    file_paths, labels = [], []
    for noise in noise_types:
        for snr in ['0dB', '5dB', '10dB']:
            path = os.path.join(data_dir, noise, snr)
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.endswith('.wav'):
                        file_paths.append(os.path.join(path, f))
                        labels.append(noise)
    return file_paths, labels

def print_dataset_stats(labels):
    print("Dataset Statistics:")
    print(f"Total samples: {len(labels)}")
    for k, v in Counter(labels).items():
        print(f"{k}: {v} samples ({100*v/len(labels):.2f}%)")

# =====================================================
# RFCC feature extraction (from training script)
# =====================================================
def make_rectangular_filterbank(sr, n_fft, n_bands):
    f_min, f_max = 0, sr/2
    freqs = np.linspace(f_min, f_max, n_bands + 1)
    fft_freqs = np.linspace(0, sr/2, int(1 + n_fft // 2))
    filters = np.zeros((n_bands, len(fft_freqs)))
    for i in range(n_bands):
        left, right = freqs[i], freqs[i+1]
        filters[i, (fft_freqs >= left) & (fft_freqs < right)] = 1.0
    return filters

def rfcc_extraction(waveform, sample_rate=8000, n_fft=512, hop_length=256,
                    n_rfcc=40, n_filter=60):
    y = waveform.squeeze().detach().cpu().numpy()
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    rect_fb = make_rectangular_filterbank(sample_rate, n_fft, n_filter)
    rect_s  = np.dot(rect_fb, S[:rect_fb.shape[1], :])
    log_rect = np.log(rect_s + 1e-10)
    rfcc = librosa.feature.mfcc(S=log_rect, n_mfcc=n_rfcc)
    return torch.tensor(rfcc, dtype=torch.float32)

def audio_transform(waveform):
    return rfcc_extraction(waveform).unsqueeze(0)  # Add channel dimension

# =====================================================
# Dataset
# =====================================================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_length=16128):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        waveform = pad_truncate_waveform(waveform, self.target_length, sr)
        return audio_transform(waveform), self.labels_encoded[idx]

# =====================================================
# Model (must match RFCC training)
# =====================================================
class NoiseClassifierCNN(nn.Module):
    def __init__(self, num_classes=3, num_conv_layers=6):
        super().__init__()
        channels = [16, 32, 64, 128, 256, 512][:num_conv_layers]
        layers, in_ch = [], 1
        for i, out_ch in enumerate(channels):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d((2,2) if i < 3 else (1,2))
            ]
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((4,4))
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(channels[-1] * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        if x.dim() == 5: x = x.squeeze(2)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        f = torch.relu(self.fc1(x))
        f = self.dropout(f)
        h = torch.relu(self.fc2(f))
        o = self.dropout(h)
        logits = self.fc3(o)
        return (logits, h) if return_features else logits

# =====================================================
# Evaluation helpers
# =====================================================
def extract_representations(model, loader, device):
    model.eval()
    labels_all, preds_all, probs_all, feats_all = [], [], [], []
    with torch.no_grad():
        for x, lbl in loader:
            x, lbl = x.to(device), lbl.to(device)
            out, feat = model(x, return_features=True)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            labels_all.extend(lbl.cpu().numpy())
            preds_all.extend(pred.cpu().numpy())
            probs_all.extend(prob.cpu().numpy())
            feats_all.extend(feat.cpu().numpy())
    return (np.array(labels_all), np.array(preds_all),
            np.array(probs_all), np.array(feats_all))

def plot_representations(features, labels, preds, method="tsne"):
    reducer = TSNE(n_components=2, random_state=42) if method == "tsne" else PCA(n_components=2)
    reduced = reducer.fit_transform(features)
    plt.figure(figsize=(8,6))
    for i, noise in enumerate(noise_types):
        mask = labels == i
        plt.scatter(reduced[mask,0], reduced[mask,1], label=noise, alpha=0.6)
    mis = labels != preds
    plt.scatter(reduced[mis,0], reduced[mis,1], c="red", marker="x", label="Misclassified")
    plt.legend()
    return plt

# =====================================================
# Main
# =====================================================
def main(args):
    file_paths, labels = load_data(args.data_dir)
    print_dataset_stats(labels)
    dataset = AudioDataset(file_paths, labels, target_length=16128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NoiseClassifierCNN(num_classes=len(noise_types),
                               num_conv_layers=args.layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    lbl_np, pred_np, prob_np, feat_np = extract_representations(model, loader, device)

    all_labels_bin = label_binarize(lbl_np, classes=list(range(len(noise_types))))
    auc_scores, calib_errors = {}, {}
    for i, n in enumerate(noise_types):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], prob_np[:, i])
        auc_scores[n] = auc(fpr, tpr)
        calib_errors[n] = brier_score_loss(all_labels_bin[:, i], prob_np[:, i])

    save_dir = "./"
    csv_path = os.path.join(save_dir, "benchmark_metrics_RFCC.csv")
    pdf_path = os.path.join(save_dir, "benchmark_report_RFCC.pdf")
    view_path = os.path.join(save_dir, "benchmark_classification_RFCC.csv")

    # Save per-file predictions
    pd.DataFrame({
        "True Label": dataset.label_encoder.inverse_transform(lbl_np),
        "Predicted Label": dataset.label_encoder.inverse_transform(pred_np),
        "Probabilities": [list(p) for p in prob_np]
    }).to_csv(view_path, index=False)

    # PDF report
    with PdfPages(pdf_path) as pdf:
        # Confusion Matrix
        cm = confusion_matrix(lbl_np, pred_np)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=noise_types).plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix"); pdf.savefig(); plt.close()

        # ROC Curves
        plt.figure()
        for i, n in enumerate(noise_types):
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], prob_np[:, i])
            plt.plot(fpr, tpr, label=f"{n} (AUC={auc_scores[n]:.2f})")
        plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.title("ROC Curve")
        pdf.savefig(); plt.close()

        # Calibration curves
        plt.figure()
        for i, n in enumerate(noise_types):
            pt, pp = calibration_curve(all_labels_bin[:, i], prob_np[:, i], n_bins=10)
            plt.plot(pp, pt, marker="o", label=n)
        plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.title("Calibration Curve")
        pdf.savefig(); plt.close()

        # Misclassifications table
        mis_idx = np.where(lbl_np != pred_np)[0][:10]
        mis_table = [[i, noise_types[lbl_np[i]], noise_types[pred_np[i]], 
                      np.round(prob_np[i], 3)] for i in mis_idx]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.axis("off")
        ax.table(cellText=mis_table,
                 colLabels=["Idx","True","Pred","Probabilities"],
                 loc="center")
        plt.title("Sample Misclassifications")
        pdf.savefig(); plt.close()

        # t-SNE & PCA
        tsne_plot = plot_representations(feat_np, lbl_np, pred_np, method="tsne")
        plt.title("t-SNE Representation"); pdf.savefig(tsne_plot.gcf()); plt.close()
        pca_plot = plot_representations(feat_np, lbl_np, pred_np, method="pca")
        plt.title("PCA Representation"); pdf.savefig(pca_plot.gcf()); plt.close()

    print(f"Benchmark report saved to {pdf_path}")
    print(f"Classification view CSV saved to {view_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pre-trained RFCC noise classifier")
    parser.add_argument("--data_dir", type=str, default="noisezus_30speakers")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="noise_classifier_RFCC_6layers_best.pth")
    parser.add_argument("--layers", type=int, default=6, choices=[3,4,5,6])
    args = parser.parse_args()
    main(args)
