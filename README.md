# Music Genre Classification Fine-Tuning

This repository provides a framework to fine-tune a pretrained VGGish model for music genre classification. The model is enhanced with an additional classifier layer, enabling you to adapt the architecture to your own dataset and custom classes. The current implementation uses the GTZAN dataset (10 genres), but you can fine-tune the model with any dataset by following the provided data format. This project is designed to process the entire audio file rather than just the first 30 seconds. Many music classification methods only analyze a fixed snippet—often the opening 30 seconds—because it's computationally simpler and the assumption is that the beginning captures the song's essence. However, this approach can overlook important musical elements that occur later in the track, such as instrumental solos, bridges, or dynamic changes that are critical for accurate genre classification. By splitting the full audio into smaller segments and aggregating the predictions, our method captures the complete range of musical content. This results in a more robust and representative feature extraction, ultimately leading to improved classification performance and a more reliable understanding of the song's overall genre.


## Overview

This project fine-tunes a pretrained VGGish model by adding a classifier layer on top. The VGGish model extracts audio features from mel spectrograms, and the additional classifier converts these embeddings into genre predictions. Although the provided code is configured for the GTZAN dataset (which includes 10 genres: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock), you can easily modify the number of classes and labels to suit your own music dataset.

## Dataset

The dataset should be prepared as a CSV file (e.g., `gtzan_labels.csv`) that contains two columns:
- **filename:** The full path to the audio file.
- **genre:** The genre label for the corresponding audio file.


## Requirements

You can install the necessary libraries from requirements.txt.
Code tested under:
python: 3.10.16
os: Ubuntu 22.04.3 LTS

## Dataset

The dataset should be prepared as a CSV file (e.g., `gtzan_labels.csv`) that contains two columns:
- **filename:** The full path to the audio file.
- **genre:** The genre label for the corresponding audio file.

### Example CSV Format

```csv
filename,genre
/gtzan_dataset/blues/track01.wav,blues
/gtzan_dataset/classical/track02.wav,classical
```


## Usage
### Data Preparation
Ensure your dataset is organized in genre-specific subdirectories, and then run the provided script (or create your own) to generate a CSV file listing the full path to each audio file along with its genre label.

### Training
This repository provides a framework to fine-tune a pretrained VGGish model for music genre classification. The model is enhanced with an additional classifier layer, enabling you to adapt the architecture to your own dataset and custom classes. The current implementation uses the GTZAN dataset (10 genres), but you can fine-tune the model with any dataset by following the provided data format. To fine-tune the model with your dataset, use the train.py script. You can customize hyperparameters (e.g., batch size, number of epochs, learning rate) via command-line arguments.

python train.py --data_dir path/to/gtzan_labels.csv --model_save_path models/vggish_genre_classifier.pth --num_epochs 30 --batch_size 8


### Inference
For inference, you can use the infer.py script with the saved model weights. This script processes an input audio file and outputs the predicted genre along with the probability distribution across genres.

python infer.py --audio_file path/to/your_audio.mp3 --model_path models/vggish_genre_classifier.pth


## Customization
Classes:
To fine-tune the model on your own dataset with different classes, simply update the list of classes (genres) in the code and ensure your CSV file reflects these labels. The CSV file must contain two columns (filename and genre). Adjust your data preparation script accordingly if your dataset is organized differently.
