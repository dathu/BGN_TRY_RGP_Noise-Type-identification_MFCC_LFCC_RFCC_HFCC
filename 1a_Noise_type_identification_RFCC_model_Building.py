'''
sudo /home/raghu/raghu/bgnenv/bin/python3.10 1a_Noise_type_identification_RFCC_24102025_Reduced_randomness_PDF_model_Building.py --layers 6 --epochs 50 --specaugment
'''
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
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
import torch.nn.functional as F
import random
import hashlib

# ==========================
# Set Random Seed for Reproducibility
# ==========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

# Always set the seed first thing
set_seed(42)

# Disable multithread nondeterminism
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Disable Tensor Cores for exact reproducibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Force deterministic CuDNN kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Limit to one CUDA stream to avoid asynchronous nondeterminism
torch.cuda.manual_seed_all(42)
try:
    torch.cuda.synchronize()
except Exception:
    pass


# ==========================
# Define the noise types
# ==========================
noise_types = ['Car', 'Exhibition', 'Station']   #Babble


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
    file_paths = []
    labels = []
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
    print("Class distribution:")
    for noise, count in class_counts.items():
        print(f"{noise}: {count} samples ({100 * count / len(labels):.2f}%)")


def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


# ==========================
# RFCC Extraction Function
# ==========================

def make_rectangular_filterbank(sr, n_fft, n_bands):
    """
    Create a linearly spaced rectangular filterbank.
    Each filter is a rectangular band-pass filter (flat inside, 0 outside).
    """
    f_min, f_max = 0, sr / 2
    freqs = np.linspace(f_min, f_max, n_bands + 1)  # band edges

    fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    filters = np.zeros((n_bands, len(fft_freqs)))

    for i in range(n_bands):
        left, right = freqs[i], freqs[i + 1]
        filters[i, (fft_freqs >= left) & (fft_freqs < right)] = 1.0  # flat response

    return filters


def rfcc_extraction(waveform, sample_rate=8000, n_fft=512, hop_length=256, n_rfcc=13, n_filter=40):
    """
    Extract RFCC (Rectangular Filter Cepstral Coefficients).
    """
    y = waveform.squeeze().detach().cpu().numpy()

    # Power spectrum
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2

    # Rectangular filterbank
    rect_fb = make_rectangular_filterbank(sample_rate, n_fft, n_filter)

    # Apply filterbank (ensure shapes align)
    rect_s = np.dot(rect_fb, S[:rect_fb.shape[1], :])

    # Log compression
    log_rect = np.log(rect_s + 1e-10)

    # DCT → RFCC
    rfcc = librosa.feature.mfcc(S=log_rect, n_mfcc=n_rfcc)

    return torch.tensor(rfcc, dtype=torch.float32)

# ==========================
# tsne and pca
# ==========================

def extract_representations(model, dataloader, device):
    model.eval()
    features, labels, preds, probs = [], [], [], []
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs, lbls = inputs.to(device), lbls.to(device)
            logits, hidden = model(inputs, return_features=True)
            features.append(hidden.cpu().numpy())
            labels.append(lbls.cpu().numpy())
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())  # ✅ add probabilities
    features = np.vstack(features)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    probs = np.vstack(probs)   # stack probabilities into 2D array
    return labels, preds, probs, features


def plot_representations(features, labels, preds, method="tsne", color_by="labels"):
    """
    Visualize feature representations with t-SNE or PCA.
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(features)

    # Choose coloring target
    if color_by == "preds":
        target = preds
        title_suffix = "Predicted Labels"
    else:
        target = labels
        title_suffix = "True Labels"

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=target, cmap="tab10", alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"{method.upper()} projection ({title_suffix})")
    return plt


# ==========================
# Custom Dataset with SpecAugment (deterministic per-sample)
# ==========================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment=False, target_length=16128, use_specaugment=True):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.target_length = target_length
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)

        self.pitch_shift = transforms.PitchShift(sample_rate=8000, n_steps=0) if augment else None
        self.use_specaugment = use_specaugment
        if augment and use_specaugment:
            self.time_mask = transforms.TimeMasking(time_mask_param=30)
            self.freq_mask = transforms.FrequencyMasking(freq_mask_param=13)
        else:
            self.time_mask = None
            self.freq_mask = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        if self.augment:
            rng = np.random.default_rng(42 + idx)
            if self.pitch_shift:
                pitch_steps = rng.uniform(-1, 1)
                waveform = transforms.PitchShift(sample_rate=8000, n_steps=pitch_steps)(waveform)
            waveform = waveform + torch.randn_like(waveform) * 0.005  # fixed noise

        waveform = pad_truncate_waveform(waveform, self.target_length, sample_rate)

        if self.transform:
            features = self.transform(waveform)
            if features.dim() > 3:
                features = features.squeeze(0)

            if self.augment and self.use_specaugment:
                # Deterministic SpecAugment: make masking reproducible per sample
                rng = np.random.default_rng(42 + idx)
                torch.manual_seed(int(rng.integers(0, 10000)))  # reseed PyTorch for this sample
                if self.time_mask:
                    features = transforms.TimeMasking(time_mask_param=30)(features)
                if self.freq_mask:
                    features = transforms.FrequencyMasking(freq_mask_param=13)(features)
        else:
            features = waveform

        label = self.labels_encoded[idx]
        return features, label


# ==========================
# Enhanced CNN Model with SE-Block, Residuals, and Tuned Parameters
# ==========================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class NoiseClassifierCNN(nn.Module):
    def __init__(self, num_classes=4, num_conv_layers=6):
        super(NoiseClassifierCNN, self).__init__()
        channels = [16, 32, 64, 128, 256, 512][:num_conv_layers]
        layers = []
        in_channels = 1

        for i, out_channels in enumerate(channels):
            kernel_size = 5 if i == 0 else 3
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            bn = nn.BatchNorm2d(out_channels, momentum=0.0, track_running_stats=False)
            relu = nn.ReLU(inplace=True)

            pool = nn.MaxPool2d((2, 2)) if i < 3 else nn.MaxPool2d((1, 2))

            block = nn.Sequential(conv, bn, relu, pool)
            layers.append(block)
            in_channels = out_channels

        self.conv_layers = nn.ModuleList(layers)
        self.se_blocks = nn.ModuleList([SEBlock(ch) for ch in channels])

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(channels[-1] * 4 * 4, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        if x.dim() == 5:
            x = x.squeeze(2)

        for conv_block, se in zip(self.conv_layers, self.se_blocks):
            residual = x
            x = conv_block(x)
            x = se(x)
            if x.shape == residual.shape:
                x = x + residual

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.norm1(self.fc1(x))))
        hidden = self.dropout(F.relu(self.norm2(self.fc2(x))))
        logits = self.fc3(hidden)

        if return_features:
            return logits, hidden
        return logits


# ==========================
# Feature Extraction Replacement
# ==========================
def audio_transform(waveform):
    rfcc = rfcc_extraction(waveform, sample_rate=8000, n_fft=512, hop_length=256, n_rfcc=40, n_filter=60)
    return rfcc.unsqueeze(0)  # Add channel dim for CNN


# ==========================
# Training Script (updated to follow LFCC flow)
# ==========================
def main(args):
    save_dir = "/home/raghu/raghu/BGN_Sir_23082025/Noizeus"
    # Ensure reproducibility for every run
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y%m%d")

    specaug_tag = "specaug" if args.specaugment else "nospecaug"
    specaug_status = "ON" if args.specaugment else "OFF"
    csv_path = os.path.join(save_dir, f"metrics_RFCC_{args.layers}layers_{specaug_tag}_{current_date}.csv")
    pdf_path = os.path.join(save_dir, f"report_RFCC_{args.layers}layers_{specaug_tag}_{current_date}.pdf")
    view_path = os.path.join(save_dir, f"classification_view_RFCC_{args.layers}layers_{specaug_tag}_{current_date}.csv")

    file_paths, labels = load_data(args.data_dir)
    print_dataset_stats(labels)
    if not file_paths:
        print("No audio files found.")
        return

    train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = AudioDataset(train_paths, train_labels, transform=audio_transform, augment=True,
                                 target_length=16128, use_specaugment=args.specaugment)
    val_dataset = AudioDataset(val_paths, val_labels, transform=audio_transform, augment=False,
                               target_length=16128, use_specaugment=False)

    g = torch.Generator()
    g.manual_seed(42)

    def seed_worker(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Training loop starting at {datetime.now()}")
    model = NoiseClassifierCNN(num_classes=len(noise_types), num_conv_layers=args.layers).to(device)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ==========================
    # Optimizer, Criterion, and Scheduler (Improved)
    # ==========================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_acc, patience_counter = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    # ➕ Add these new variables
    best_labels, best_preds, best_probs = [], [], []

    # ==========================
    # Training Loop (with gradient clipping)
    # ==========================
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            # ✅ Save best epoch predictions and probabilities
            best_labels = all_labels.copy()
            best_preds = all_preds.copy()
            best_probs = all_probs.copy()

            # ✅ Save best model weights
            torch.save(model.state_dict(), os.path.join(save_dir, f"noise_classifier_RFCC_{args.layers}layers_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # ==========================
    # Reproducibility Verification
    # ==========================
    def model_checksum(model):
        """Compute a deterministic checksum for all model parameters."""
        hash_md5 = hashlib.md5()
        for param in model.parameters():
            hash_md5.update(param.detach().cpu().numpy().tobytes())
        return hash_md5.hexdigest()

    checksum = model_checksum(model)
    print(f"Model checksum (for reproducibility check): {checksum}")


    # -----------------------
    # keep the same CSV, PDF, visualization code (but use best_* saved above)
    # -----------------------

    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    all_labels_bin = label_binarize(best_labels, classes=list(range(len(noise_types))))
    all_probs = np.array(best_probs)
    auc_scores, calib_errors = {}, {}
    for i, noise in enumerate(noise_types):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        auc_scores[noise] = auc(fpr, tpr)
        calib_errors[noise] = brier_score_loss(all_labels_bin[:, i], all_probs[:, i])

    summary = {"epoch": "final", "train_loss": np.min(history["train_loss"]),
               "val_loss": np.min(history["val_loss"]), "val_accuracy": best_val_acc,
               "SpecAugment": args.specaugment}
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    df_view = pd.DataFrame({
        "True Label": val_dataset.label_encoder.inverse_transform(best_labels),
        "Predicted Label": val_dataset.label_encoder.inverse_transform(best_preds),
        "Probabilities": [list(p) for p in best_probs]
    })
    df_view.to_csv(view_path, index=False)

    # PDF Report
    with PdfPages(pdf_path) as pdf:
        # Loss Curve
        plt.figure()
        plt.plot(df["epoch"][:-1], df["train_loss"][:-1], label="Train Loss")
        plt.plot(df["epoch"][:-1], df["val_loss"][:-1], label="Validation Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Loss Curve ({args.layers} layers, SpecAug={specaug_status})")
        plt.legend(); pdf.savefig(); plt.close()

        # Accuracy Curve
        plt.figure()
        plt.plot(df["epoch"][:-1], df["val_accuracy"][:-1], label="Validation Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy Curve ({args.layers} layers, SpecAug={specaug_status})")
        plt.legend(); pdf.savefig(); plt.close()

        # Confusion Matrix
        cm = confusion_matrix(best_labels, best_preds)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=noise_types).plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix ({args.layers} layers, SpecAug={specaug_status})")
        pdf.savefig(); plt.close()

        # ROC Curve
        plt.figure()
        for i, noise in enumerate(noise_types):
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            plt.plot(fpr, tpr, label=f"{noise} (AUC={auc_scores[noise]:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({args.layers} layers, SpecAug={specaug_status})")
        plt.legend(loc="lower right"); pdf.savefig(); plt.close()

        # Calibration Curve
        plt.figure()
        for i, noise in enumerate(noise_types):
            prob_true, prob_pred = calibration_curve(all_labels_bin[:, i], all_probs[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=f"{noise}")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Curve ({args.layers} layers, SpecAug={specaug_status})")
        plt.legend(); pdf.savefig(); plt.close()

        # Misclassifications
        mis_idx = np.where(np.array(best_labels) != np.array(best_preds))[0][:10]
        mis_table = [[i, noise_types[best_labels[i]], noise_types[best_preds[i]], np.round(best_probs[i], 3)]
                     for i in mis_idx]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.table(cellText=mis_table, colLabels=["Idx", "True", "Pred", "Probabilities"], loc="center")
        plt.title(f"Sample Misclassifications (SpecAug={specaug_status})")
        pdf.savefig(); plt.close()

        # t-SNE & PCA
        labels_np, preds_np, probs_np, features_np = extract_representations(model, val_loader, device)
        # t-SNE and PCA plots (both true and predicted labels)
        for method in ["tsne", "pca"]:
            # True labels
            plot_true = plot_representations(features_np, labels_np, preds_np, method=method, color_by="labels")
            plt.title(f"{method.upper()} Representation (True Labels, {args.layers} layers, SpecAug={specaug_status})")
            pdf.savefig(plot_true.gcf()); plt.close()

            # Predicted labels
            plot_pred = plot_representations(features_np, labels_np, preds_np, method=method, color_by="preds")
            plt.title(f"{method.upper()} Representation (Predicted Labels, {args.layers} layers, SpecAug={specaug_status})")
            pdf.savefig(plot_pred.gcf()); plt.close()



    print(f"Metrics CSV saved to {csv_path}")
    print(f"Classification view CSV saved to {view_path}")
    print(f"PDF report saved to {pdf_path}")


# ==========================
# CLI Entry Point
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise Type Identification with CNN + SpecAugment toggle (RFCC version)")
    parser.add_argument("--data_dir", type=str, default=".", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--layers", type=int, default=6, choices=[3, 4, 5, 6], help="Number of CNN layers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--specaugment", action="store_true", help="Enable SpecAugment (time/freq masking)")
    args = parser.parse_args()
    main(args)
