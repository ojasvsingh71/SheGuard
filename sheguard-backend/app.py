from flask import Flask, request, jsonify
from flask import make_response
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask_cors import CORS, cross_origin
import logging

app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class SimpleDeepfakeDetector(nn.Module):
    """Simple CNN-based deepfake detector"""
    def __init__(self):
        super(SimpleDeepfakeDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Real vs Fake
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeepfakeDetectionService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self._load_model()
    
    def _load_model(self):
        """Load the deepfake detection model with fallback options"""
        try:
            # Try to load a pre-trained model first
            self._load_pretrained_model()
        except Exception as e:
            logger.warning(f"Failed to load pre-trained model: {e}")
            # Fallback to simple model
            self._load_simple_model()
    
    def _load_pretrained_model(self):
        """Try to load a pre-trained model from Hugging Face"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Try a different, more reliable model
            model_name = "dima806/deepfake_vs_real_image_detection"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.hf_model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model_type = "huggingface"
            logger.info("Successfully loaded Hugging Face model")
            
        except Exception as e:
            logger.warning(f"Hugging Face model failed: {e}")
            raise e
    
    def _load_simple_model(self):
        """Load a simple CNN model as fallback"""
        self.model = SimpleDeepfakeDetector()
        self.model.eval()
        self.model_type = "simple"
        logger.info("Loaded simple CNN model as fallback")
    
    def detect_faces(self, image_path):
        """Detect faces in the image using OpenCV"""
        try:
            # Load the image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return len(faces) > 0, len(faces)
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return False, 0
    
    def analyze_image_quality(self, image_path):
        """Analyze image quality metrics that might indicate manipulation"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate image entropy
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            return {
                'blur_score': float(laplacian_var),
                'entropy': float(entropy),
                'is_blurry': laplacian_var < 100
            }
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return {'blur_score': 0, 'entropy': 0, 'is_blurry': False}
    
    def predict_with_hf_model(self, image_path):
        """Predict using Hugging Face model"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=1).item()
            
            # Get confidence score
            confidence = float(probabilities[0][predicted_class])
            
            # Map prediction to label
            labels = self.hf_model.config.id2label
            prediction = labels[predicted_class]
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"HF model prediction error: {e}")
            raise e
    
    def predict_with_simple_model(self, image_path):
        """Predict using simple CNN model (fallback)"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            confidence = float(probabilities[0][predicted_class])
            prediction = "REAL" if predicted_class == 0 else "FAKE"
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Simple model prediction error: {e}")
            return "UNKNOWN", 0.5
    
    def analyze_image(self, image_path):
        """Main analysis function"""
        try:
            # Check if faces are present
            has_faces, face_count = self.detect_faces(image_path)
            
            # Analyze image quality
            quality_metrics = self.analyze_image_quality(image_path)
            
            # Get deepfake prediction
            if self.model_type == "huggingface":
                try:
                    prediction, confidence = self.predict_with_hf_model(image_path)
                except:
                    prediction, confidence = self.predict_with_simple_model(image_path)
            else:
                prediction, confidence = self.predict_with_simple_model(image_path)
            
            # Combine all analysis
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'has_faces': has_faces,
                'face_count': face_count,
                'quality_metrics': quality_metrics,
                'model_used': self.model_type
            }
            
            # Add risk assessment
            risk_factors = []
            if not has_faces:
                risk_factors.append("No faces detected")
            if quality_metrics['is_blurry']:
                risk_factors.append("Image appears blurry")
            if confidence < 0.7:
                risk_factors.append("Low confidence prediction")
            
            result['risk_factors'] = risk_factors
            result['risk_level'] = 'HIGH' if len(risk_factors) >= 2 else 'MEDIUM' if len(risk_factors) == 1 else 'LOW'
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'model_used': self.model_type
            }

# Initialize the detection service
detector = DeepfakeDetectionService()

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/')
@cross_origin()
def home():
    return jsonify({
        "message": "SheGuard Backend is running!",
        "model_status": detector.model_type,
        "version": "2.0"
    })

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if 'file' not in request.files:
        logger.error("No file part in the request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected.")
        return jsonify({"error": "No selected file"}), 400

    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload an image file."}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    logger.info(f"Saving file to: {file_path}")
    
    try:
        file.save(file_path)
        return jsonify({
            "message": "File uploaded successfully", 
            "file_path": file_path,
            "filename": file.filename
        })
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return jsonify({"error": "Failed to save file"}), 500

@app.route('/analyze', methods=['POST'])
@cross_origin()
def analyze():
    file_path = request.json.get('file_path')
    if not file_path:
        logger.error("File path is missing.")
        return jsonify({"error": "File path is required"}), 400

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({"error": "File not found"}), 404

    logger.info(f"Analyzing file: {file_path}")
    
    try:
        result = detector.analyze_image(file_path)
        logger.info(f"Analysis complete: {result['prediction']} (confidence: {result.get('confidence', 0):.2f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return jsonify({
            "error": "Analysis failed",
            "details": str(e),
            "prediction": "ERROR"
        }), 500

@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({
        "status": "healthy",
        "model_type": detector.model_type,
        "device": str(detector.device)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)