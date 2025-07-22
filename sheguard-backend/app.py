from flask import Flask, request, jsonify
from flask import make_response
import os
import cv2
import numpy as np
from PIL import Image
import logging
from flask_cors import CORS, cross_origin
import base64
import io

app = Flask(__name__)

# Configure CORS properly
CORS(app, 
     origins=["http://localhost:3000", "https://sheguard-frontend.onrender.com", "*"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
     supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class LightweightDeepfakeDetector:
    """Lightweight deepfake detector using traditional computer vision techniques"""
    
    def __init__(self):
        self.face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """Load OpenCV face detector"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("Face detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
    
    def detect_faces(self, image_path):
        """Detect faces in the image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, 0, []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return len(faces) > 0, len(faces), faces.tolist()
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return False, 0, []
    
    def analyze_image_quality(self, image_path):
        """Analyze image quality metrics"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate image statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
                'blur_score': float(laplacian_var),
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(std_brightness),
                'edge_density': float(edge_density),
                'is_blurry': laplacian_var < 100,
                'is_dark': mean_brightness < 50,
                'is_bright': mean_brightness > 200,
                'low_contrast': std_brightness < 30
            }
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return {'error': str(e)}
    
    def detect_compression_artifacts(self, image_path):
        """Detect JPEG compression artifacts that might indicate manipulation"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT to detect blocking artifacts
            h, w = gray.shape
            block_size = 8
            artifact_score = 0
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    dct_block = cv2.dct(block)
                    # High frequency components indicate artifacts
                    high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                    artifact_score += high_freq
            
            artifact_score /= ((h // block_size) * (w // block_size))
            
            return {
                'compression_score': float(artifact_score),
                'has_artifacts': artifact_score > 1000
            }
        except Exception as e:
            logger.error(f"Compression analysis error: {e}")
            return {'error': str(e)}
    
    def heuristic_deepfake_detection(self, image_path, faces, quality_metrics):
        """Use heuristic rules to detect potential deepfakes"""
        try:
            # Initialize suspicion score
            suspicion_score = 0.0
            reasons = []
            
            # Face-related checks
            if len(faces) == 0:
                suspicion_score += 0.3
                reasons.append("No faces detected")
            elif len(faces) > 3:
                suspicion_score += 0.2
                reasons.append("Multiple faces detected")
            
            # Quality checks
            if quality_metrics.get('is_blurry', False):
                suspicion_score += 0.2
                reasons.append("Image appears blurry")
            
            if quality_metrics.get('low_contrast', False):
                suspicion_score += 0.1
                reasons.append("Low contrast image")
            
            # Brightness checks
            if quality_metrics.get('is_dark', False) or quality_metrics.get('is_bright', False):
                suspicion_score += 0.1
                reasons.append("Unusual brightness levels")
            
            # Edge density check
            edge_density = quality_metrics.get('edge_density', 0)
            if edge_density < 0.05:  # Very few edges might indicate smoothing
                suspicion_score += 0.2
                reasons.append("Unusually smooth image")
            elif edge_density > 0.3:  # Too many edges might indicate artifacts
                suspicion_score += 0.1
                reasons.append("High edge density")
            
            # Face size consistency (if multiple faces)
            if len(faces) > 1:
                face_areas = [(w * h) for (x, y, w, h) in faces]
                if len(set([int(area/1000) for area in face_areas])) > 1:  # Different sizes
                    suspicion_score += 0.15
                    reasons.append("Inconsistent face sizes")
            
            # Determine prediction based on suspicion score
            if suspicion_score >= 0.6:
                prediction = "FAKE"
                confidence = min(0.9, 0.5 + suspicion_score)
            elif suspicion_score >= 0.3:
                prediction = "SUSPICIOUS"
                confidence = 0.6
            else:
                prediction = "REAL"
                confidence = min(0.9, 0.9 - suspicion_score)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'suspicion_score': suspicion_score,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Heuristic detection error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_image(self, image_path):
        """Main analysis function"""
        try:
            # Detect faces
            has_faces, face_count, faces = self.detect_faces(image_path)
            
            # Analyze image quality
            quality_metrics = self.analyze_image_quality(image_path)
            
            # Detect compression artifacts
            compression_analysis = self.detect_compression_artifacts(image_path)
            
            # Heuristic deepfake detection
            detection_result = self.heuristic_deepfake_detection(image_path, faces, quality_metrics)
            
            # Combine all analysis
            result = {
                'prediction': detection_result['prediction'],
                'confidence': detection_result['confidence'],
                'has_faces': has_faces,
                'face_count': face_count,
                'faces': faces,
                'quality_metrics': quality_metrics,
                'compression_analysis': compression_analysis,
                'detection_reasons': detection_result.get('reasons', []),
                'suspicion_score': detection_result.get('suspicion_score', 0),
                'model_used': 'heuristic_cv'
            }
            
            # Add risk assessment
            risk_factors = []
            if not has_faces:
                risk_factors.append("No faces detected")
            if quality_metrics.get('is_blurry', False):
                risk_factors.append("Image appears blurry")
            if detection_result['confidence'] < 0.7:
                risk_factors.append("Low confidence prediction")
            if compression_analysis.get('has_artifacts', False):
                risk_factors.append("Compression artifacts detected")
            
            result['risk_factors'] = risk_factors
            
            # Determine risk level
            if len(risk_factors) >= 3 or detection_result['suspicion_score'] > 0.6:
                result['risk_level'] = 'HIGH'
            elif len(risk_factors) >= 1 or detection_result['suspicion_score'] > 0.3:
                result['risk_level'] = 'MEDIUM'
            else:
                result['risk_level'] = 'LOW'
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'model_used': 'heuristic_cv'
            }

# Initialize the detection service
detector = LightweightDeepfakeDetector()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Access-Control-Allow-Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/')
def home():
    return jsonify({
        "message": "SheGuard Lightweight Backend is running!",
        "model_status": "heuristic_cv",
        "version": "3.0-lightweight",
        "memory_optimized": True
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_type": "heuristic_cv",
        "memory_usage": "optimized",
        "face_detector": "loaded" if detector.face_cascade is not None else "failed"
    })

@app.route('/upload', methods=['POST'])
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)