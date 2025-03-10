
# Music Genre Classification

## Overview
This project leverages a pretrained VGGish model as a feature extractor, adding a custom classifier on top to perform music genre classification. The approach uses full audio processing by splitting each audio file into 1-second segments, processing them individually, and then aggregating predictions to capture the overall musical content.

This project uses the GTZAN dataset, a widely used benchmark for music genre classification. The GTZAN dataset contains 1,000 audio clips (each 30 seconds long) distributed evenly across 10 genres:

Blues
Classical
Country
Disco
Hiphop
Jazz
Metal
Pop
Reggae
Rock

For this project, it was organized the audio files in a directory structure where each genre has its own folder. A helper script convert_gtzan_format.py is provided to generate a CSV file (gtzan_labels.csv) that lists the full path to each audio file along with its corresponding genre label. This CSV file is then used by our data loader.

Folder structure should be like that:
music-genre-classification/ ├── data/ │ └── gtzan_labels.csv # CSV file with audio file paths and genre labels ├── models/ │ └── vggish_genre_classifier.pth # Saved model weights (best model) ├── src/ │ ├── train.py # Training script │ ├── infer.py # Inference script that uses full audio processing ├── README.md ├── requirements.txt


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/music-genre-classification.git
   cd music-genre-classification

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


pip install -r requirements.txt

training:
python src/train.py --data_dir data/gtzan_labels.csv --model_save_path models/vggish_genre_classifier.pth --num_epochs 30 --batch_size 8

inference:
python src/infer_audio.py --audio_file path/to/your_audio.mp3 --model_path models/vggish_genre_classifier.pth
