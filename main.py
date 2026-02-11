import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.multiprocessing as mp
import torchaudio
import torchvision
import warnings
from pathlib import Path

# Parameters for loading audio files and creating spectrograms
SPECTROGRAM_DPI = 200
SAMPLE_RATE = 44100
SOUND_PATH = "./data/sounds"
IMAGE_SIZE = (1024, 1024)
CHANNELS = 3

# AI model parameter definitions
IMAGE_PATH = "./data/images"
ACCURACY_THRESHOLD = 0.90

# Warning handling
warnings.filterwarnings("ignore")

def load_audio_files() -> list[str]:
    print("Loading audio files from dataset...")
    audio_files = []
    root = Path(SOUND_PATH).resolve()
    extensions = ["*.wav", "*.mp3", "*.flac"]
    for ext in extensions:
        for filepath in root.rglob(ext):
            audio_files.append(str(filepath))

    return audio_files

def create_spectrogram(audio_path: str) -> None:
    try:
        plot_path = os.path.join(IMAGE_PATH, Path(os.path.basename(audio_path)).stem + ".png")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.numpy()
        _, axes = plt.subplots(1, 1)
        axes.specgram(waveform[0], Fs=sample_rate)
        plt.axis('off')
        plt.savefig(plot_path, dpi=SPECTROGRAM_DPI, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating spectrogram for {audio_path}: {e}")

def main():
    audio_files = load_audio_files()
    print(f"Loaded {len(audio_files)} audio files.")
    for audio_file in audio_files:
        create_spectrogram(audio_file)

if __name__ == "__main__":
    main()
