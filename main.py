import torch
import torch.nn as nn
import torchaudio
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# parameters for loading audio files and creating spectrograms
SOUND_PATH = "./data"
SAMPLE_RATE = 16000
WINDOW_SIZE = 1.0  # in seconds
OVERLAP = 0.5  # fraction of overlap between windows
N_MELS = 64
TIME_FRAMES = 128
BATCH_SIZE = 32
EPOCHS = 10

# warning handling
warnings.filterwarnings("ignore")

class SoundDataset(Dataset):
    def __init__(self, root, window_size=WINDOW_SIZE, overlap=OVERLAP):
        # obtain list of audio files and prepare spectrogram transforms
        self.root = Path(root)
        self.files = [file for ext in ["*.wav", "*.mp3", "*.flac"] for file in self.root.rglob(ext)]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.window_samples = int(window_size * SAMPLE_RATE)
        self.hop_samples = int((1 - overlap) * self.window_samples)
        self.index = []

        # create label map and index of windows
        classes = sorted({file.parent.name for file in self.files})
        self.label_map = {cls_name: i for i, cls_name in enumerate(classes)}

        for filepath in self.files:
            try:
                file_label = filepath.parent.name
                waveform, sample_rate = torchaudio.load(filepath)

                # resample if sample rate is different from target
                if sample_rate != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)

                # convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                total_samples = waveform.shape[1]

                # split into overlapping windows of fixed size
                for start in range(0, total_samples - self.window_samples, self.hop_samples):
                    self.index.append((filepath, start, file_label))

            # if exception detected (e.g., corrupted file), skip to next file
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        try:
            filepath, start, label_name = self.index[idx]
            waveform, sample_rate = torchaudio.load(filepath)

            # resample if sample rate is different from target
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = waveform[:, start:start + self.window_samples]
            
            # create log mel spectrogram
            spectrogram = self.mel(waveform)
            spectrogram = self.db(spectrogram)

            # add padding to spectrogram if shorter than expected
            current_frames = spectrogram.shape[2]
            if current_frames < TIME_FRAMES:
                padding = TIME_FRAMES - current_frames
                spectrogram = nn.functional.pad(spectrogram, (0, padding))
            else:
                spectrogram = spectrogram[:, :, :TIME_FRAMES]

            # normalize spectrogram
            spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
            label_name = filepath.parent.name
            label = self.label_map[label_name]

            return spectrogram, label
        
        except Exception as e:
            print(f"Error processing {filepath} at index {idx}: {e}")
            return torch.zeros(1, N_MELS, TIME_FRAMES), -1  # return dummy data for failed samples
        
    def files_count(self):
        return len(self.files)
    
class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def load_audio_files() -> SoundDataset:
    print(f"Loading audio files from dataset in {SOUND_PATH}...")
    dataset = SoundDataset(SOUND_PATH, window_size=WINDOW_SIZE, overlap=OVERLAP)
    return dataset

def train(dataset: SoundDataset, dataloader: DataLoader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpectrogramCNN(num_classes=len(dataset.label_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for spectrograms, labels in dataloader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss/len(dataloader)}")

def main():
    audio_files = load_audio_files()
    print(f"Loaded {audio_files.files_count()} audio files.")
    print(f"Total samples in dataset: {len(audio_files)}")

    dataloader = DataLoader(audio_files, batch_size=BATCH_SIZE, shuffle=True)
    for spectrogram, labels in dataloader:
        print(f"Spectrogram batch shape: {spectrogram.shape}, Labels batch shape: {labels.shape}")
        
    train(audio_files, dataloader)

if __name__ == "__main__":
    main()
