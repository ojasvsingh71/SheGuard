from flask import Flask, request, jsonify
import os
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained deepfake detection model from Hugging Face
model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")

@app.route('/')
def home():
    return "SheGuard Backend is running!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("No file selected.")
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(f"Saving file to: {file_path}")
    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "file_path": file_path})

@app.route('/analyze', methods=['POST'])
def analyze():
    file_path = request.json.get('file_path')
    if not file_path:
        print("File path is missing.")
        return jsonify({"error": "File path is required"}), 400

    print(f"Analyzing file: {file_path}")
    try:
        # Load the image
        image = Image.open(file_path).convert("RGB")
        print("Image loaded successfully.")

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        print("Image preprocessed successfully.")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Get the predicted label
        label = model.config.id2label[predicted_class]
        print(f"Prediction: {label}")

        return jsonify({
            "prediction": label
        })
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)