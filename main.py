import torch
import torch.nn as nn
import torchaudio
import warnings
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparameters and Constants
SOUND_PATH  = "./data"
SAMPLE_RATE = 16000
WINDOW_SIZE = 1.0        # seconds
OVERLAP     = 0.5        # fraction of overlap between windows
N_MELS      = 64
TIME_FRAMES = 128
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
VAL_SPLIT   = 0.15       # 15 % of windows used for validation
TEST_SPLIT  = 0.15       # 15 % of windows used for final test
SEED        = 42
MODEL_PATH  = "fish_classifier.pt"
warnings.filterwarnings("ignore")

# Reproducibility 
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dataset
class SoundDataset(Dataset):
    """
    Indexes all overlapping windows across a folder tree, where each
    sub-folder name is treated as a class label.

    Files are NOT loaded during __init__; only metadata is collected so
    that the index step remains fast even for large datasets.
    """

    def __init__(self, root: str, window_size: float = WINDOW_SIZE,
                 overlap: float = OVERLAP, augment: bool = False):
        self.root = Path(root)
        self.augment = augment

        # Spectrogram transforms
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=512,
            n_fft=1024,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # SpecAugment — applied only when augment=True
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

        self.window_samples = int(window_size * SAMPLE_RATE)
        self.hop_samples    = int((1 - overlap) * self.window_samples)

        # Discover audio files and build label map
        self.files = [
            f for ext in ("*.wav", "*.mp3", "*.flac")
            for f in self.root.rglob(ext)
        ]
        if not self.files:
            raise FileNotFoundError(f"No audio files found under '{root}'.")

        classes = sorted({f.parent.name for f in self.files})
        self.label_map = {cls: i for i, cls in enumerate(classes)}
        print(f"Classes detected: {self.label_map}")

        # Build window index
        # Each entry: (filepath, start_sample, label_int)
        # Note: `+ 1` ensures the last full window is included.
        self.index: list[tuple[Path, int, int]] = []

        for filepath in self.files:
            try:
                waveform, orig_sr = torchaudio.load(filepath)
                num_frames = waveform.shape[1]
                total_samples = int(num_frames * SAMPLE_RATE / orig_sr)
                label = self.label_map[filepath.parent.name]

                windows = list(range(
                    0,
                    total_samples - self.window_samples + 1,
                    self.hop_samples,
                ))
                print(
                    f"[DEBUG] {filepath.name} | "
                    f"duration={num_frames / orig_sr:.2f}s | "
                    f"resampled_samples={total_samples} | "
                    f"windows={len(windows)}"
                )
                for start in windows:
                    self.index.append((filepath, start, label))

            except Exception as e:
                print(f"[WARN] Skipping {filepath}: {e}")

    # Dunder helpers
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        filepath, start, label = self.index[idx]

        try:
            waveform, sr = torchaudio.load(filepath)

            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=SAMPLE_RATE
                )
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Slice the requested window; pad if the file is too short
            clip = waveform[:, start : start + self.window_samples]
            if clip.shape[1] < self.window_samples:
                clip = nn.functional.pad(
                    clip, (0, self.window_samples - clip.shape[1])
                )

            # Log-mel spectrogram
            spec = self.to_db(self.mel(clip))  # (1, n_mels, time)

            # Fix time dimension
            t = spec.shape[2]
            if t < TIME_FRAMES:
                spec = nn.functional.pad(spec, (0, TIME_FRAMES - t))
            else:
                spec = spec[:, :, :TIME_FRAMES]

            # Per-sample standardisation (guard against silent clips)
            mean = spec.mean()
            std  = spec.std()
            spec = (spec - mean) / (std + 1e-6)

            # SpecAugment (training only)
            if self.augment:
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)

            return spec, label

        except Exception as e:
            print(f"[ERROR] {filepath} @ {start}: {e}")
            # Return a clearly invalid label so a custom collate_fn can
            # filter it out.
            return torch.zeros(1, N_MELS, TIME_FRAMES), -1

    # Convenience
    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def file_count(self) -> int:
        return len(self.files)


def collate_fn(batch):
    """Drop any samples that failed to load (label == -1)."""
    batch = [(s, l) for s, l in batch if l != -1]
    if not batch:
        return None
    specs, labels = zip(*batch)
    return torch.stack(specs), torch.tensor(labels, dtype=torch.long)


# Model
class SpectrogramCNN(nn.Module):
    """
    Lightweight CNN for log-mel spectrogram classification.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),   # fixed spatial size regardless of input
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


# Training & evaluation
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            specs, labels = batch
            specs, labels = specs.to(device), labels.to(device)

            outputs = model(specs)
            total_loss += criterion(outputs, labels).item()
            correct   += (outputs.argmax(dim=1) == labels).sum().item()
            total     += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train(train_loader: DataLoader, val_loader: DataLoader,
          num_classes: int) -> nn.Module:
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = SpectrogramCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    print(f"\nTraining on {device}\n{'─' * 50}")

    for epoch in range(1, EPOCHS + 1):
        # Training pass
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            if batch is None:
                continue
            specs, labels = batch
            specs, labels = specs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total+= labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2%}  │  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2%}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    # Reload best weights before returning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model


# Augmented subset wrapper
class AugmentedSubset(Dataset):
    """
    Wraps a random_split Subset and temporarily enables augmentation on the
    underlying SoundDataset for each __getitem__ call.
    Must be defined at module level so DataLoader workers can pickle it.
    """
    def __init__(self, subset):
        self.subset  = subset
        self.dataset = subset.dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx  = self.subset.indices[idx]
        orig_flag = self.dataset.augment
        self.dataset.augment = True
        item = self.dataset[real_idx]
        self.dataset.augment = orig_flag
        return item


# Entry point
def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and split dataset
    full_dataset = SoundDataset(SOUND_PATH, augment=False)
    print(f"Audio files : {full_dataset.file_count}")
    print(f"Total windows: {len(full_dataset)}")
    print(f"Classes : {full_dataset.label_map}\n")

    n_total = len(full_dataset)

    if n_total == 0:
        raise RuntimeError(
            f"Dataset is empty"
        )

    # Guarantee every split has at least 1 sample
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_test  = max(1, int(n_total * TEST_SPLIT))
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise RuntimeError(
            f"Dataset too small to split: {n_total} window(s) total. "
        )

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        AugmentedSubset(train_set), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    # Training
    model = train(train_loader, val_loader, num_classes=full_dataset.num_classes)

    # Final test evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'─' * 50}")
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()