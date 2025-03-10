import glob
import pandas as pd

data_path = "/mnt/c/Users/furka/Desktop/music_genre_classification/gtzan_dataset"

items = glob.glob(data_path + "/*/*.wav")

data = []

for i in items:
    data.append([i, i.split("/")[-2]])

df = pd.DataFrame(data, columns=["filename", "genre"])
df.to_csv("gtzan_labels.csv", index=False)