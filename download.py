import os
import gdown
import zipfile

# Create folders if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- 1. Download YOLO model ---
print("Downloading YOLO model...")
gdown.download("https://drive.google.com/uc?id=15feuOZuRX5n1xIusi47G_KsEzdVtAs7N", "model/best.pt", quiet=False)

# --- 2. Download CSV dataset ---
print("Downloading CSV...")
gdown.download("https://drive.google.com/uc?id=1amgX3-tohS9ho2WFWcbfXGTRGZ7ubEap", "dataset/Updated_furniture_recommendation_dataset.csv", quiet=False)

# --- 3. Download and unzip dataset ZIP ---
print("Downloading zipped dataset (train/valid)...")
gdown.download_folder(id="1zJwKWsGHXTCFwpRk5KuVdCFmL2pQbe4U", quiet=False, use_cookies=False, output="dataset")

# --- 4. Download data.yaml ---
print("Downloading data.yaml...")
gdown.download("https://drive.google.com/uc?id=1-H5GtZFlKOgD0BGTDLQIUsUcLY15GYp6", "data/data.yaml", quiet=False)

print("✅ All files downloaded.")