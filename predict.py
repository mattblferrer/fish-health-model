import sys
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# Must match the values used during training
SAMPLE_RATE  = 16000
WINDOW_SIZE  = 1.0
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


# Preprocessing (must match __getitem__ in training)
def load_spectrogram(filepath: str) -> torch.Tensor:
    mel    = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=512, n_fft=1024
    )
    to_db  = torchaudio.transforms.AmplitudeToDB(top_db=80)

    waveform, sr = torchaudio.load(filepath)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Use the first window only (you could average all windows for longer clips)
    window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
    clip = waveform[:, :window_samples]
    if clip.shape[1] < window_samples:
        clip = nn.functional.pad(clip, (0, window_samples - clip.shape[1]))

    spec = to_db(mel(clip))

    t = spec.shape[2]
    if t < TIME_FRAMES:
        spec = nn.functional.pad(spec, (0, TIME_FRAMES - t))
    else:
        spec = spec[:, :, :TIME_FRAMES]

    # Normalize
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)

    return spec.unsqueeze(0)  # add batch dimension -> (1, 1, n_mels, time)


# Inference 
def predict(filepath: str, output_file: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = SpectrogramCNN(num_classes=len(LABEL_MAP)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Preprocess audio
    spec = load_spectrogram(filepath).to(device)

    # Run inference
    with torch.no_grad():
        outputs    = model(spec)
        probs      = torch.softmax(outputs, dim=1)[0]  # convert logits to probabilities
        confidence, predicted_idx = probs.max(dim=0)
        label      = LABEL_MAP[predicted_idx.item()]

    print(f"File       : {Path(filepath).name}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence.item():.1%}")
    print()
    for idx, class_name in LABEL_MAP.items():
        print(f"  {class_name:<12} {probs[idx].item():.1%}")

    with open(output_file, "a") as f:
        f.write(f"{Path(filepath).name},{label},{confidence.item():.4f}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py path/to/audio.wav output.csv")
        print("   or: python predict.py path/to/folder/ output.csv")
        sys.exit(1)

    target = Path(sys.argv[1])
    output_file = sys.argv[2]

    # Write CSV header
    with open(output_file, "a") as f:
        f.write("filename,prediction,confidence\n")

    if target.is_dir():
        # Predict on all audio files in a folder
        files = [f for ext in ("*.wav", "*.mp3", "*.flac") for f in target.rglob(ext)]
        if not files:
            print(f"No audio files found in {target}")
            sys.exit(1)
        for f in files:
            predict(str(f), output_file)
    else:
        predict(str(target), output_file)
    