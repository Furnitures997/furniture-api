import os
import cv2
import random
import base64
import numpy as np
import yaml
import pandas as pd
from flask import Flask, request, jsonify, session
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "abdo"

MODEL_PATH = "model/best.pt"
DATASET_CSV = "dataset/Updated_furniture_recommendation_dataset.csv"
YAML_PATH = "data/data.yaml"
DATASET_PATH = "dataset"

with open(YAML_PATH, 'r') as f:
    dataset_config = yaml.safe_load(f)
CLASS_LABELS = dataset_config['names']

model = YOLO(MODEL_PATH)

df = pd.read_csv(DATASET_CSV)
df['Specific Requests'] = df['Specific Requests'].fillna('No')
df['Recommended Furniture'] = df['Recommended Furniture'].apply(lambda x: eval(x) if isinstance(x, str) else x)
df['Recommended Furniture'] = df['Recommended Furniture'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

feature_columns = [
    'Primary Purpose', 'Design Style', 'Furniture Type', 'Room Size',
    'Dimension Constraints', 'Budget', 'Storage Needs'
]
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[feature_columns])

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except:
        return None

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def encode_image_to_base644(image):
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
                    images = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(('.jpg', '.png'))]
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
        if len(similar_images) >= top_n:
            break
    return similar_images[:top_n]

def get_recommendations(user_input, top_n=5):
    user_df = pd.DataFrame([user_input])
    user_encoded = encoder.transform(user_df[feature_columns])
    user_similarities = cosine_similarity(user_encoded, encoded_features)
    top_indices = user_similarities.argsort()[0][-top_n:][::-1]
    recommended_furniture = df.iloc[top_indices]["Recommended Furniture"].tolist()
    return {"recommendations": recommended_furniture}

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_preferences = {
        "Primary Purpose": data.get("Primary_Purpose"),
        "Design Style": data.get("Design_Style"),
        "Furniture Type": data.get("Furniture_Type"),
        "Room Size": data.get("Room_Size"),
        "Dimension Constraints": data.get("Dimension_Constraints"),
        "Budget": data.get("Budget"),
        "Storage Needs": data.get("Storage_Needs"),
    }
    recommendations = get_recommendations(user_preferences)
    label_list = recommendations["recommendations"][0].split(", ")
    similar_images = get_similar_images(label_list)
    if not similar_images:
        return jsonify({"error": "No images found for the given labels"}), 404
    session["remaining_images"] = similar_images[1:]
    img_base64 = encode_image_to_base64(similar_images[0])
    return jsonify({"image": img_base64})

@app.route('/next_recommendation', methods=['POST'])
def next_recommendation():
    remaining_images = session.get("remaining_images", [])
    if not remaining_images:
        return jsonify({"error": "No more recommendations available"}), 404
    next_image_path = remaining_images.pop(0)
    session["remaining_images"] = remaining_images
    img_base64 = encode_image_to_base64(next_image_path)
    return jsonify({"image": img_base64})

@app.route('/objectdetection', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    extracted_objects = []
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(class_id)]
        cropped_image = image[y1:y2, x1:x2]
        cropped_image_base64 = encode_image_to_base644(cropped_image)
        extracted_objects.append({"label": class_name, "image": cropped_image_base64})
    return jsonify({"objects": extracted_objects})

@app.route('/objectdetectionrec', methods=['POST'])
def predict_rec():
    data = request.json
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
    image_data = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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
        cropped_image_base64 = encode_image_to_base644(cropped_image)
        extracted_objects.append({"label": class_name, "image": cropped_image_base64})
    return jsonify({"objects": extracted_objects})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')