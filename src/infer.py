import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import librosa
from torchaudio.prototype.pipelines import VGGISH

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)

class GenreClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.base(x)  
        x = self.classifier(x)
        return x

def process_full_audio(file_path, sample_rate=16000, n_mels=64):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
    except Exception as e:
        logging.error("Error loading audio file: %s", e)
        sys.exit(1)
    
    segment_length = sample_rate  
    segments = []
    
    for start in range(0, len(audio), segment_length):
        segment = audio[start: start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_fft=400, hop_length=160, win_length=400, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        tensor_segment = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
        segments.append(tensor_segment)
    
    return segments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/c/Users/furka/Desktop/music_genre_classification/vggish_genre_classifier.pth")
    parser.add_argument("--audio_file", type=str, default="/mnt/c/Users/furka/Desktop/music_genre_classification/Stabil - Ben Kimim(MP3_160K).mp3")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()



    device = torch.device(args.device)
    logging.info("Using device: %s", device)

    logging.info("Loading pretrained VGGish model.")
    vggish = VGGISH.get_model()

    logging.info("Initializing GenreClassifier model.")
    model = GenreClassifier(vggish, num_classes=NUM_CLASSES)
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        logging.error("Error loading model state dictionary: %s", e)
        sys.exit(1)
    
    model.to(device)
    model.eval()

    logging.info("Processing full audio file: %s", args.audio_file)
    segments = process_full_audio(args.audio_file, sample_rate=16000, n_mels=64)
    if not segments:
        logging.error("No segments were extracted from the audio file.")
        sys.exit(1)

    all_probs = []
    with torch.no_grad():
        for seg in segments:
            # Add a batch dimension to each segment: (1, 1, n_mels, T)
            seg_input = seg.unsqueeze(0).to(device)
            output = model(seg_input)  # shape: (1, num_classes)
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
            all_probs.append(probs)
    
    # Aggregate predictions by averaging probabilities over segments
    all_probs = np.array(all_probs)  # shape: (num_segments, num_classes)
    mean_probs = all_probs.mean(axis=0)
    predicted_idx = int(mean_probs.argmax())
    predicted_genre = GENRES[predicted_idx]

    logging.info("Predicted Genre: %s", predicted_genre)
    print("Aggregated Genre Probabilities:")
    for genre, prob in zip(GENRES, mean_probs):
        print(f"  {genre}: {prob:.4f}")

if __name__ == "__main__":
    main()
