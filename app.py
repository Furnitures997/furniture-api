# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VkREqNGmp94tWxeNUeSzK10zlRA3YhKU
"""

import os
import cv2
import random
from flask import Flask, request, jsonify, session
import base64
import numpy as np
import yaml
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import psutil

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']  # Must be set in Render dashboard

# ===== KEY CHANGE 2: Update all file paths =====
# Define relative paths (Render will look in the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")          # Path to YOLO model
DATASET_PATH = os.path.join(BASE_DIR, "data")                        # Dataset folder
YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")                   # YOLO config
CSV_PATH = os.path.join(DATASET_PATH, "Updated_furniture_recommendation_dataset.csv") # Recommendation data

import torch
torch.set_num_threads(1)

# Load the YOLO model
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])  # Whitelist YOLO's model class

print("Loading Model.")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# Load class labels from YAML
with open(YAML_PATH, 'r') as f:
    dataset_config = yaml.safe_load(f)
CLASS_LABELS = dataset_config['names']

# Load recommendation dataset
df = pd.read_csv(CSV_PATH)
df['Specific Requests'] = df['Specific Requests'].fillna('No')
df['Recommended Furniture'] = df['Recommended Furniture'].apply(lambda x: eval(x) if isinstance(x, str) else x)
df['Recommended Furniture'] = df['Recommended Furniture'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Feature columns for recommendation
feature_columns = [
    'Primary Purpose', 'Design Style', 'Furniture Type', 'Room Size',
    'Dimension Constraints', 'Budget', 'Storage Needs'
]

# One-Hot Encoder
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[feature_columns])

# Helper functions
def encode_image_to_base64(image):
    if image is None or not isinstance(image, np.ndarray):
        return None
    _, img_encoded = cv2.imencode('.jpg', image)
    return base64.b64encode(img_encoded).decode('utf-8')

def get_similar_images(labels, top_n=10):
    similar_images = []
    for label in labels:
        if label in CLASS_LABELS:
            label_index = CLASS_LABELS.index(label)
            for folder in ["train", "valid"]:
                images_folder = os.path.join(DATASET_PATH, folder, "images")
                labels_folder = os.path.join(DATASET_PATH, folder, "labels")
                if os.path.exists(images_folder) and os.path.exists(labels_folder):
                    images = [os.path.join(images_folder, img) for img in os.listdir(images_folder)
                             if img.endswith(('.jpg', '.png'))]
                    random.shuffle(images)
                    for img_path in images:
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        annotation_path = os.path.join(labels_folder, f"{img_name}.txt")
                        if os.path.exists(annotation_path):
                            with open(annotation_path, "r") as f:
                                lines = f.readlines()
                            for line in lines:
                                if int(line.split()[0]) == label_index:
                                    similar_images.append(img_path)
                                    break
                        if len(similar_images) >= top_n:
                            break
                    if len(similar_images) >= top_n:
                        break
    return similar_images[:top_n]

def get_recommendations(user_input, top_n=5):
    user_input_with_defaults = user_input.copy()
    user_input_with_defaults["Room Size"] = "More than 20 sqm"  # Set a constant value
    user_input_with_defaults["Dimension Constraints"] = "No"  # Set a constant value
    user_df = pd.DataFrame([user_input_with_defaults])
    user_encoded = encoder.transform(user_df[feature_columns])
    user_similarities = cosine_similarity(user_encoded, encoded_features)
    top_indices = user_similarities.argsort()[0][-top_n:][::-1]
    recommended_furniture = df.iloc[top_indices]["Recommended Furniture"].tolist()
    return {"recommendations": recommended_furniture}

# Routes
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_preferences = {
        "Primary Purpose": data.get("Primary_Purpose"),
        "Design Style": data.get("Design_Style"),
        "Furniture Type": data.get("Furniture_Type"),
        "Budget": data.get("Budget"),
        "Storage Needs": data.get("Storage_Needs"),
    }
    recommendations = get_recommendations(user_preferences)
    recommended_labels = recommendations["recommendations"][0].split(", ")
    similar_images = get_similar_images(recommended_labels)

    if not similar_images:
        return jsonify({"error": "No images found for the given labels"}), 404

    session["remaining_images"] = similar_images[1:]
    img_base64 = encode_image_to_base64(cv2.imread(similar_images[0]))

    return jsonify({"image": img_base64})

@app.route('/next_recommendation', methods=['POST'])
def next_recommendation():
    remaining_images = session.get("remaining_images", [])
    if not remaining_images:
        return jsonify({"error": "No more recommendations available"}), 404

    next_image_path = remaining_images.pop(0)
    session["remaining_images"] = remaining_images
    img_base64 = encode_image_to_base64(cv2.imread(next_image_path))

    return jsonify({"image": img_base64})

@app.route('/objectdetection', methods=['POST'])
def predict():
    print("Request Sent.")
    process = psutil.Process(os.getpid())
    print("Memory before:", process.memory_info().rss / 1024 / 1024, "MB")
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    print("File Uploaded.")
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image, save=False, device='cpu')
    print("Memory after:", process.memory_info().rss / 1024 / 1024, "MB")
    print(results)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    print(class_names)
    extracted_objects = []

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(class_id)]
        cropped_image = image[y1:y2, x1:x2]
        cropped_image_base64 = encode_image_to_base64(cropped_image)
        extracted_objects.append({"label": class_name, "image": cropped_image_base64})

    return jsonify({"objects": extracted_objects})

@app.route('/objectdetectionrec', methods=['POST'])
def predict_rec():
    data = request.json
    image_base64 = data.get("image")

    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    image_data = base64.b64decode(image_base64)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid base64 image"}), 400

    results = model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    extracted_objects = []

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(class_id)]
        cropped_image = image[y1:y2, x1:x2]
        cropped_image_base64 = encode_image_to_base64(cropped_image)
        extracted_objects.append({"label": class_name, "image": cropped_image_base64})

    return jsonify({"objects": extracted_objects})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
