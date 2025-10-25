import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import librosa

# ==========================
# Define the noise types
# ==========================
noise_types = ['Car', 'Exhibition', 'Station']

# ==========================
# Utility Functions
# ==========================
def pad_truncate_waveform(waveform, target_length, sample_rate):
    current_length = waveform.shape[-1]
    if current_length > target_length:
        waveform = waveform[..., :target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
    return waveform

def load_data(data_dir='.'):
    file_paths, labels = [], []
    for noise in noise_types:
        noise_dir = os.path.join(data_dir, noise)
        for snr_dir in ['0dB', '5dB', '10dB']:
            snr_path = os.path.join(noise_dir, snr_dir)
            if os.path.exists(snr_path):
                for file in os.listdir(snr_path):
                    if file.endswith('.wav'):
                        file_paths.append(os.path.join(snr_path, file))
                        labels.append(noise)
    return file_paths, labels

def print_dataset_stats(labels):
    print("Dataset Statistics:")
    print(f"Total samples: {len(labels)}")
    class_counts = Counter(labels)
    for noise, count in class_counts.items():
        print(f"{noise}: {count} samples ({100 * count / len(labels):.2f}%)")

# ==========================
# HFCC Extraction
# ==========================
def hz_to_erb(f): return 21.4 * np.log10(1 + 0.00437 * f)
def erb_to_hz(erb): return (10**(erb / 21.4) - 1) / 0.00437

def make_erb_filterbank(sr, n_fft, n_bands):
    f_min, f_max = 0, sr/2
    erb_min, erb_max = hz_to_erb(f_min), hz_to_erb(f_max)
    erb_points = np.linspace(erb_min, erb_max, n_bands + 2)
    freqs = erb_to_hz(erb_points)
    fft_freqs = np.linspace(0, sr/2, int(1 + n_fft // 2))
    filters = np.zeros((n_bands, len(fft_freqs)))
    for i in range(1, n_bands + 1):
        left, center, right = freqs[i-1], freqs[i], freqs[i+1]
        left_slope = (fft_freqs - left) / (center - left)
        right_slope = (right - fft_freqs) / (right - center)
        filters[i-1] = np.maximum(0, np.minimum(left_slope, right_slope))
    return filters

def hfcc_extraction(waveform, sample_rate=8000, n_fft=512, hop_length=256, n_hfcc=13):
    y = waveform.squeeze().detach().cpu().numpy()
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    erb_fb = make_erb_filterbank(sample_rate, n_fft, n_hfcc)
    erb_s = np.dot(erb_fb, S[:erb_fb.shape[1], :])
    log_erb = np.log(erb_s + 1e-10)
    hfcc = librosa.feature.mfcc(S=log_erb, n_mfcc=n_hfcc)
    return torch.tensor(hfcc, dtype=torch.float32)

def audio_transform(waveform):
    hfcc = hfcc_extraction(waveform, sample_rate=8000, n_fft=512, hop_length=256, n_hfcc=13)
    return hfcc.unsqueeze(0)

# ==========================
# Dataset
# ==========================
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None, target_length=16128):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_length = target_length
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        waveform = pad_truncate_waveform(waveform, self.target_length, sample_rate)
        features = self.transform(waveform) if self.transform else waveform
        label = self.labels_encoded[idx]
        return features, label

# ==========================
# Model (must match training)
# ==========================
class NoiseClassifierCNN(nn.Module):
    def __init__(self, num_classes=3, num_conv_layers=3):
        super().__init__()
        channels = [16, 32, 64][:num_conv_layers]
        layers, in_channels = [], 1
        for out_channels in channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ]
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(channels[-1] * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        if x.dim() == 5: x = x.squeeze(2)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        features = torch.relu(self.fc1(x))
        features = self.dropout(features)
        hidden = torch.relu(self.fc2(features))
        out = self.dropout(hidden)
        logits = self.fc3(out)
        return (logits, hidden) if return_features else logits

# ==========================
# Benchmark Evaluation
# ==========================
def extract_representations(model, data_loader, device):
    model.eval()
    all_labels, all_preds, all_probs, all_features = [], [], [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, features = model(inputs, return_features=True)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_features)

def plot_representations(features, labels, preds, method="tsne"):
    reducer = TSNE(n_components=2, random_state=42) if method == "tsne" else PCA(n_components=2)
    reduced = reducer.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for i, noise in enumerate(noise_types):
        mask = labels == i
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=noise, alpha=0.6)
    mis_mask = labels != preds
    plt.scatter(reduced[mis_mask, 0], reduced[mis_mask, 1], c="red", marker="x", label="Misclassified")
    plt.legend()
    return plt

# ==========================
# Main
# ==========================
def main(args):
    file_paths, labels = load_data(args.data_dir)
    print_dataset_stats(labels)
    dataset = AudioDataset(file_paths, labels, transform=audio_transform, target_length=16128)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NoiseClassifierCNN(num_classes=len(noise_types), num_conv_layers=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    labels_np, preds_np, probs_np, features_np = extract_representations(model, data_loader, device)

    all_labels_bin = label_binarize(labels_np, classes=list(range(len(noise_types))))
    auc_scores, calib_errors = {}, {}
    for i, noise in enumerate(noise_types):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], probs_np[:, i])
        auc_scores[noise] = auc(fpr, tpr)
        calib_errors[noise] = brier_score_loss(all_labels_bin[:, i], probs_np[:, i])

    save_dir = "./"
    csv_path = os.path.join(save_dir, "benchmark_metrics_HFCC.csv")
    pdf_path = os.path.join(save_dir, "benchmark_report_HFCC.pdf")
    view_path = os.path.join(save_dir, "benchmark_classification_HFCC.csv")

    df_view = pd.DataFrame({"True Label": dataset.label_encoder.inverse_transform(labels_np),
                            "Predicted Label": dataset.label_encoder.inverse_transform(preds_np),
                            "Probabilities": [list(p) for p in probs_np]})
    df_view.to_csv(view_path, index=False)

    # PDF
    with PdfPages(pdf_path) as pdf:
        # Confusion Matrix
        cm = confusion_matrix(labels_np, preds_np)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=noise_types).plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        pdf.savefig(); plt.close()

        # ROC
        plt.figure()
        for i, noise in enumerate(noise_types):
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], probs_np[:, i])
            plt.plot(fpr, tpr, label=f"{noise} (AUC={auc_scores[noise]:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.legend(); plt.title("ROC Curve")
        pdf.savefig(); plt.close()

        # Calibration
        plt.figure()
        for i, noise in enumerate(noise_types):
            prob_true, prob_pred = calibration_curve(all_labels_bin[:, i], probs_np[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label=noise)
        plt.plot([0,1],[0,1],"k--")
        plt.legend(); plt.title("Calibration Curve")
        pdf.savefig(); plt.close()

        # Misclassifications
        mis_idx = np.where(labels_np != preds_np)[0][:10]
        mis_table = [[i, noise_types[labels_np[i]], noise_types[preds_np[i]], np.round(probs_np[i],3)] for i in mis_idx]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.axis("off")
        ax.table(cellText=mis_table, colLabels=["Idx","True","Pred","Probabilities"], loc="center")
        plt.title("Sample Misclassifications")
        pdf.savefig(); plt.close()

        # t-SNE & PCA
        tsne_plot = plot_representations(features_np, labels_np, preds_np, method="tsne")
        plt.title("t-SNE Representation"); pdf.savefig(tsne_plot.gcf()); plt.close()
        pca_plot = plot_representations(features_np, labels_np, preds_np, method="pca")
        plt.title("PCA Representation"); pdf.savefig(pca_plot.gcf()); plt.close()

    print(f"Benchmark report saved to {pdf_path}")
    print(f"Classification view CSV saved to {view_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pre-trained HFCC noise classifier")
    parser.add_argument("--data_dir", type=str, default="noisezus_30speakers", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--model_path", type=str, default="noise_classifier_3layers_best_HFCC.pth", help="Path to trained model")
    args = parser.parse_args()
    main(args)

