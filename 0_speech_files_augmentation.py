import os
import librosa
import soundfile as sf
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, BandStopFilter, ClippingDistortion

# Input and output directories
input_dir = "Station"  #Replace with Car, Exhibition, Station
output_dir = "Station_augmented" #Replace with Car, Exhibition, Station

# -------------------------------
# Define augmentation pipelines
# -------------------------------

'''
1. Time-Domain Augmentation
Time shifting: Slightly shift the audio forward/backward (e.g., by a few milliseconds).
Speed perturbation: Speed up or slow down the playback (e.g., 0.9x, 1.1x) without changing pitch.
Pitch shifting: Change pitch slightly (±2 semitones).
Volume scaling: Increase/decrease loudness.

'''

# 1. Time-Domain Augmentation
time_domain_augment = Compose([
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.5)
])

'''
3. Spectrogram-Domain Augmentation
SpecAugment (widely used in speech recognition):
Randomly mask frequency bands.
Randomly mask time segments.
This forces the model to learn more robust features.
'''

# 3. "Spectrogram"-like Augmentation (using filters & distortions)
spectrogram_augment = Compose([
    BandStopFilter(
        min_center_freq=200, max_center_freq=4000,
        min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
        p=0.7
    ),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40, p=0.5)
])

'''
5. Combination Techniques
Apply multiple augmentations together (e.g., time shift + pitch shift + noise overlay).
'''

# 5. Combination Techniques
combination_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
    BandStopFilter(
        min_center_freq=200, max_center_freq=4000,
        min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
        p=0.5
    )
])

'''
4. Cropping & Padding
Split longer files into multiple shorter clips.
Randomly crop sections of audio to create new samples.
Pad shorter clips with silence/noise.
'''

# -------------------------------
# Cropping & Padding
# -------------------------------
def crop_or_pad(audio, sr, target_duration=2.0):
    target_length = int(sr * target_duration)
    if len(audio) > target_length:
        start = np.random.randint(0, len(audio) - target_length)
        return audio[start:start + target_length]
    else:
        return np.pad(audio, (0, max(0, target_length - len(audio))), "constant")

# -------------------------------
# Augmentation Loop
# -------------------------------
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            # Load audio
            audio, sr = librosa.load(file_path, sr=None)

            # --- 1. Time-Domain Augmentation ---
            aug_audio = time_domain_augment(samples=audio, sample_rate=sr)
            sf.write(os.path.join(output_subdir, f"time_{file}"), aug_audio, sr)

            # --- 3. Spectrogram-like Augmentation ---
            aug_audio = spectrogram_augment(samples=audio, sample_rate=sr)
            sf.write(os.path.join(output_subdir, f"spec_{file}"), aug_audio, sr)

            # --- 4. Cropping & Padding ---
            cropped_audio = crop_or_pad(audio, sr, target_duration=2.0)
            sf.write(os.path.join(output_subdir, f"crop_{file}"), cropped_audio, sr)

            # --- 5. Combination Techniques ---
            aug_audio = combination_augment(samples=audio, sample_rate=sr)
            sf.write(os.path.join(output_subdir, f"combo_{file}"), aug_audio, sr)

print(" Augmentation complete! Augmented files saved in:", output_dir)
