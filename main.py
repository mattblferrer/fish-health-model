import torch.nn as nn
import torchaudio
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# parameters for loading audio files and creating spectrograms
SOUND_PATH = "./data"
SAMPLE_RATE = 16000
WINDOW_SIZE = 1.0  # in seconds
OVERLAP = 0.5      # fraction of overlap between windows
N_MELS = 64

# warning handling
warnings.filterwarnings("ignore")

class SoundDataset(Dataset):
    def __init__(self, root, window_size=WINDOW_SIZE, overlap=OVERLAP):
        try:
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
                for start in range(0, total_samples - self.window_samples + 1, self.hop_samples):
                    self.index.append((filepath, start, file_label))

            # mel + log transforms
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_mels=N_MELS
            )
            self.db = torchaudio.transforms.AmplitudeToDB()

        # if exception detected (e.g., corrupted file), skip to next file
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, start, label_name = self.index[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform[:, start:start + self.window_samples]
        
        # create log mel spectrogram
        spectrogram = self.mel(waveform)
        spectrogram = self.db(spectrogram)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        label_name = file_path.parent.name
        label = self.label_map[label_name]

        return spectrogram, label
    
    def files_count(self):
        return len(self.files)
    


def load_audio_files() -> SoundDataset:
    print(f"Loading audio files from dataset in {SOUND_PATH}...")
    dataset = SoundDataset(SOUND_PATH, window_size=WINDOW_SIZE, overlap=OVERLAP)
    return dataset

def main():
    audio_files = load_audio_files()
    print(f"Loaded {audio_files.files_count()} audio files.")
    print(f"Total samples in dataset: {len(audio_files)}")
    dataloader = DataLoader(audio_files, batch_size=32, shuffle=True)
    for spectrogram, labels in dataloader:
        print(f"Spectrogram batch shape: {spectrogram.shape}, Labels batch shape: {labels.shape}")

if __name__ == "__main__":
    main()
