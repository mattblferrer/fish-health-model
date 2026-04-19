import sys
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# Must match the values used during training
SAMPLE_RATE  = 16000
WINDOW_SIZE  = 1.0
OVERLAP      = 0.5
N_MELS       = 64
TIME_FRAMES  = 128
MODEL_PATH   = "fish_classifier.pt"

# Must match the order classes were discovered during training (alphabetical)
LABEL_MAP = {0: "no_fish_sound", 1: "stressed", 2: "unstressed"}


# Copy of the model architecture (must be identical to training)
class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))


# Preprocessing 
def load_all_windows(filepath: str) -> torch.Tensor:
    """
    Slices a full audio file into overlapping windows and returns a batch of
    spectrograms. Shape: (N, 1, n_mels, time_frames)
    Falls back to a single padded window for clips shorter than WINDOW_SIZE.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=512, n_fft=1024
    )
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=SAMPLE_RATE
        )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
    hop_samples    = int((1 - OVERLAP) * window_samples)
    total_samples  = waveform.shape[1]

    # Build window start positions; ensure at least one window for short clips
    starts = list(range(0, max(1, total_samples - window_samples + 1), hop_samples))

    windows = []
    for start in starts:
        clip = waveform[:, start : start + window_samples]

        # Pad if the final clip is shorter than a full window
        if clip.shape[1] < window_samples:
            clip = nn.functional.pad(clip, (0, window_samples - clip.shape[1]))

        spec = to_db(mel(clip))

        t = spec.shape[2]
        if t < TIME_FRAMES:
            spec = nn.functional.pad(spec, (0, TIME_FRAMES - t))
        else:
            spec = spec[:, :, :TIME_FRAMES]

        # Per-window normalisation (matches training)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        windows.append(spec)

    return torch.stack(windows)  # (N, 1, n_mels, time_frames)


# Inference
def predict(filepath: str, model: nn.Module, device: str, output_file: str) -> None:
    windows = load_all_windows(filepath).to(device)  # (N, 1, n_mels, time)

    with torch.no_grad():
        outputs   = model(windows)                  # (N, num_classes) — all windows at once
        probs     = torch.softmax(outputs, dim=1)   # (N, num_classes)
        avg_probs = probs.mean(dim=0)               # average across all windows

    confidence, predicted_idx = avg_probs.max(dim=0)
    label = LABEL_MAP[predicted_idx.item()]

    print(f"File       : {Path(filepath).name}")
    print(f"Windows    : {len(windows)}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence.item():.1%}")
    print()
    for idx, class_name in LABEL_MAP.items():
        print(f"  {class_name:<12} {avg_probs[idx].item():.1%}")
    print()

    with open(output_file, "a") as f:
        f.write(f"{Path(filepath).name},{label}")
        for idx, class_name in LABEL_MAP.items():
            f.write(f",{avg_probs[idx].item():.4f}")
        f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py path/to/audio.wav output.csv")
        print("   or: python predict.py path/to/folder/ output.csv")
        sys.exit(1)

    target      = Path(sys.argv[1])
    output_file = sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model once and reuse it for every file
    model = SpectrogramCNN(num_classes=len(LABEL_MAP)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Write CSV header
    with open(output_file, "w") as f:
        f.write("filename,prediction,")
        for idx, class_name in LABEL_MAP.items():
            f.write(f"{class_name}")
            if idx < len(LABEL_MAP) - 1:
                f.write(",")
        f.write("\n")

    if target.is_dir():
        files = [f for ext in ("*.wav", "*.mp3", "*.flac") for f in target.rglob(ext)]
        if not files:
            print(f"No audio files found in {target}")
            sys.exit(1)
        for f in files:
            predict(str(f), model, device, output_file)
    else:
        predict(str(target), model, device, output_file)