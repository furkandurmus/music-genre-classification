import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.prototype.pipelines import VGGISH

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
NUM_CLASSES = len(GENRES)

class LabeledAudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, n_mels=64):
        """
        Loads a CSV file (audio_dir) with at least two columns:
            - 'filename': full path to the audio file.
            - 'genre': genre label.
        Splits each audio file into 1‑second segments.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.labels_df = pd.read_csv(audio_dir)
        self.label_encoder = LabelEncoder()
        self.labels_df['genre_encoded'] = self.label_encoder.fit_transform(self.labels_df['genre'])
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self): 
        return len(self.labels_df)

    def __getitem__(self, idx):
        file_path = self.labels_df.iloc[idx]['filename']
        genre_label = self.labels_df.iloc[idx]['genre_encoded']
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            segment_length = self.sample_rate  # 1 second worth of samples
            segments = []
            # Split audio into non-overlapping 1-second segments
            for start in range(0, len(audio), segment_length):
                segment = audio[start: start + segment_length]
                if len(segment) < segment_length:
                    segment = np.pad(segment, (0, segment_length - len(segment)))
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=400,
                                                          hop_length=160, win_length=400, n_mels=self.n_mels)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
                tensor_segment = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
                segments.append(tensor_segment)
            segments_tensor = torch.stack(segments, dim=0)
            return segments_tensor, genre_label
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def custom_collate(batch):
    """
    Since each sample returns a tensor of shape (num_segments, 1, n_mels, T) and the number of segments
    may vary per audio file, this collate function returns:
        - A list of tensors (one per sample)
        - A tensor of labels
    """
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return inputs, labels

# Define fine-tuning model
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
        # x: (batch, 1, n_mels, T)
        # VGGish processes each segment independently and returns embeddings of shape (batch, 128)
        x = self.base(x)
        x = self.classifier(x)
        return x

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_inputs, labels in dataloader:
            # Process each audio file in the batch individually
            batch_outputs = []
            for sample in batch_inputs:
                sample = sample.to(device)  # sample shape: (num_segments, 1, n_mels, T)
                seg_outputs = model(sample)  # shape: (num_segments, num_classes)
                # Average predictions over all segments
                avg_output = seg_outputs.mean(dim=0, keepdim=True)  # shape: (1, num_classes)
                batch_outputs.append(avg_output)
            outputs = torch.cat(batch_outputs, dim=0)  # shape: (batch_size, num_classes)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = loss_total / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def train_model(args):
    # Load full dataset from CSV file
    logging.info("Loading data from: %s", args.data_dir)
    full_dataset = LabeledAudioDataset(audio_dir=args.data_dir) 
    dataset_size = len(full_dataset)
    logging.info("Total samples found: %d", dataset_size)
    
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset into training, validation, and test subsets
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=generator)
    logging.info("Train/Val/Test sizes: %d/%d/%d", train_size, val_size, test_size)
    
    # Create DataLoaders using the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    # Load pretrained VGGish model and set embeddings mode
    logging.info("Loading pretrained VGGish model.")
    vggish = VGGISH.get_model()
    vggish.embeddings = True

    # Initialize classifier model
    model = GenreClassifier(vggish, num_classes=NUM_CLASSES)
    model.to(args.device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0

    logging.info("Starting training for %d epochs on %s", args.num_epochs, args.device)
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        model.train()
        running_loss = 0.0
        for batch_inputs, labels in tqdm(train_loader):
            # For each audio file in the batch, process all segments and average predictions.
            batch_outputs = []
            for sample in batch_inputs:
                sample = sample.to(args.device)  # sample shape: (num_segments, 1, n_mels, T)
                seg_outputs = model(sample)        # shape: (num_segments, num_classes)
                avg_output = seg_outputs.mean(dim=0, keepdim=True)  # Average over segments
                batch_outputs.append(avg_output)
            outputs = torch.cat(batch_outputs, dim=0)  # shape: (batch_size, num_classes)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        logging.info("Epoch %d/%d, Training Loss: %.4f", epoch, args.num_epochs, avg_train_loss)

        # Validation step
        val_loss, val_acc = evaluate(model, val_loader, args.device)
        logging.info("Epoch %d, Validation Loss: %.4f, Accuracy: %.4f", epoch, val_loss, val_acc)

        # Save best model weights based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_save_path)
            logging.info("New best model saved with Validation Accuracy: %.4f", best_val_acc)

    # Evaluate on test set after training
    test_loss, test_acc = evaluate(model, test_loader, args.device)
    logging.info("Test Loss: %.4f, Test Accuracy: %.4f", test_loss, test_acc)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/c/Users/furka/Desktop/music_genre_classification/gtzan_labels.csv", type=str)
    parser.add_argument("--model_save_path", type=str, default="vggish_genre_classifier.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    train_model(args)
