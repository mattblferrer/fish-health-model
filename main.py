import torch.nn as nn
import torchaudio
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Parameters for loading audio files and creating spectrograms
SOUND_PATH = "./data"
SAMPLE_RATE = 16000
N_MELS = 64

# Warning handling
warnings.filterwarnings("ignore")

class SoundDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = [file for ext in ["*.wav", "*.mp3", "*.flac"] for file in self.root.rglob(ext)]
        classes = sorted({file.parent.name for file in self.files})
        self.label_map = {cls_name: i for i, cls_name in enumerate(classes)}
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path = self.files[idx]
            waveform, sample_rate = torchaudio.load(file_path)

            # convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # fix length to 1 second
            target_length = SAMPLE_RATE
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                padding = target_length - waveform.shape[1]
                waveform = nn.functional.pad(waveform, (0, padding))

            spectrogram = self.mel(waveform)
            label_name = file_path.parent.name
            label = self.label_map[label_name]

            return spectrogram, label
        
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def load_audio_files() -> SoundDataset:
    print(f"Loading audio files from dataset in {SOUND_PATH}...")
    dataset = SoundDataset(SOUND_PATH)
    return dataset

def main():
    audio_files = load_audio_files()
    print(f"Loaded {len(audio_files)} audio files.")
    dataloader = DataLoader(audio_files, batch_size=32, shuffle=True)
    for spectrogram, labels in dataloader:
        print(f"Spectrogram batch shape: {spectrogram.shape}, Labels batch shape: {labels.shape}")

if __name__ == "__main__":
    main()
