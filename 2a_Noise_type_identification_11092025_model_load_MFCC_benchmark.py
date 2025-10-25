import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, classification_report,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ==========================
# Auto-detect noise types from folder structure
# ==========================
def get_noise_types(data_dir='noisezus'):
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


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


def load_data(data_dir='noisezus'):
    file_paths, labels = [], []
    noise_types = get_noise_types(data_dir)
    for noise in noise_types:
        noise_dir = os.path.join(data_dir, noise)
        for root, _, files in os.walk(noise_dir):   # recursive scan
            for file in files:
                if file.endswith('.wav'):
                    file_paths.append(os.path.join(root, file))
                    labels.append(noise)
    return file_paths, labels, noise_types


# ==========================
# Dataset
# ==========================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, target_length=16128):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_length = target_length
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        waveform = pad_truncate_waveform(waveform, self.target_length, sample_rate)

        if self.transform:
            mel_spectrogram = self.transform(waveform)
            if mel_spectrogram.dim() > 3:
                mel_spectrogram = mel_spectrogram.squeeze(0)
        else:
            mel_spectrogram = waveform

        label = self.labels_encoded[idx]
        return mel_spectrogram, label, self.file_paths[idx]


# ==========================
# CNN Model (same as training)
# ==========================
class NoiseClassifierCNN(nn.Module):
    def __init__(self, num_classes=4, num_conv_layers=6):
        super(NoiseClassifierCNN, self).__init__()
        channels = [16, 32, 64, 128, 256, 512][:num_conv_layers]
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.6)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            conv_out = self.conv_layers(dummy)
            flatten_size = conv_out.view(1, -1).size(1)
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = torch.relu(self.fc1(x))
        features = self.dropout(features)
        hidden = torch.relu(self.fc2(features))
        out = self.dropout(hidden)
        logits = self.fc3(out)
        return logits


# ==========================
# Main Testing
# ==========================
def main():
    # Load test data
    file_paths, labels, noise_types = load_data('noisezus')
    print(f"Found {len(file_paths)} files for testing.")
    print("Detected noise types:", noise_types)
    print("Class distribution:", Counter(labels))

    mel_transform = transforms.MelSpectrogram(sample_rate=8000, n_fft=512, hop_length=256, n_mels=64)
    def audio_transform(waveform):
        return transforms.AmplitudeToDB()(mel_transform(waveform))

    test_dataset = AudioDataset(file_paths, labels, transform=audio_transform, target_length=16128)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NoiseClassifierCNN(num_classes=len(noise_types), num_conv_layers=6).to(device)
    model.load_state_dict(torch.load("noise_classifier_6layers_best.pth", map_location=device))
    model.eval()

    # Test
    all_preds, all_labels, all_files, all_probs = [], [], [], []
    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_files.extend(paths)
            all_probs.extend(probs.cpu().numpy())

    # Decode labels
    le = test_dataset.label_encoder
    true_labels = le.inverse_transform(all_labels)
    pred_labels = le.inverse_transform(all_preds)
    all_probs = np.array(all_probs)

    # Accuracy
    acc = accuracy_score(true_labels, pred_labels)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Save results CSV
    df = pd.DataFrame({
        "File": all_files,
        "True Label": true_labels,
        "Predicted Label": pred_labels
    })
    df.to_csv("test_results.csv", index=False)
    print("Results saved to test_results.csv")

    # Classification Report
    report_dict = classification_report(true_labels, pred_labels, labels=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("classification_report.csv")
    print("Classification report saved to classification_report.csv")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=le.classes_)

    # ==========================
    # PDF Report
    # ==========================
    with PdfPages("test_report.pdf") as pdf:
        # Page 1: Summary
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.1, 0.8, f"Test Accuracy: {acc*100:.2f}%", fontsize=14, weight="bold")
        plt.text(0.1, 0.7, f"Total Samples: {len(test_dataset)}", fontsize=12)
        plt.title("Noise Classification Test Summary")
        pdf.savefig(); plt.close()

        # Page 2: Per-class accuracy bar chart
        class_accuracies = []
        for i, cls in enumerate(le.classes_):
            cls_indices = np.where(true_labels == cls)[0]
            cls_correct = np.sum(np.array(pred_labels)[cls_indices] == cls)
            cls_acc = cls_correct / len(cls_indices) if len(cls_indices) > 0 else 0
            class_accuracies.append(cls_acc * 100)

        plt.figure(figsize=(8, 5))
        plt.bar(le.classes_, class_accuracies, color="skyblue")
        plt.ylabel("Accuracy (%)")
        plt.title("Per-Class Accuracy")
        for i, v in enumerate(class_accuracies):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        pdf.savefig(); plt.close()

        # Page 3: Classification Report Table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.table(cellText=report_df.values.round(3),
                 colLabels=report_df.columns,
                 rowLabels=report_df.index,
                 loc="center")
        plt.title("Classification Report")
        pdf.savefig(); plt.close()

        # Page 4: Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix - Test Data")
        pdf.savefig(); plt.close()

        # Page 5: ROC Curves
        true_labels_bin = label_binarize(all_labels, classes=np.arange(len(le.classes_)))
        plt.figure(figsize=(6, 6))
        for i, noise in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{noise} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Test Data")
        plt.legend(loc="lower right")
        pdf.savefig(); plt.close()

        # Page 6: Calibration Curves
        plt.figure(figsize=(6, 6))
        for i, noise in enumerate(le.classes_):
            prob_true, prob_pred = calibration_curve(
                true_labels_bin[:, i], all_probs[:, i], n_bins=10
            )
            plt.plot(prob_pred, prob_true, marker='o', label=noise)
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curves - Test Data")
        plt.legend()
        pdf.savefig(); plt.close()

    print("PDF report saved to test_report.pdf")


if __name__ == "__main__":
    main()

