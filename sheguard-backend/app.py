from flask import Flask, request, jsonify
from flask import make_response
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from flask_cors import CORS, cross_origin
import base64
import io
from scipy import ndimage
from skimage import feature, measure, filters
import math
from advanced_detector import UltraPrecisionDeepfakeDetector

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


# Initialize the ultra-precision detection service
detector = UltraPrecisionDeepfakeDetector()

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
        "message": "SheGuard Ultra-Precision Backend is running!",
        "model_status": "ultra_precision_ensemble_cv",
        "version": "5.0-ultra-precision",
        "features": [
            "Ultra-comprehensive facial analysis",
            "Multi-method symmetry detection",
            "Advanced texture analysis with GLCM & Gabor filters",
            "Comprehensive eye & iris analysis",
            "Advanced skin authenticity detection",
            "Micro-expression analysis",
            "Lighting consistency verification",
            "Frequency domain analysis",
            "Biometric consistency checking",
            "Ultra-precision ensemble scoring"
        ]
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_type": "ultra_precision_ensemble_cv",
        "detectors": {
            "face_detector": "loaded" if detector.face_cascade is not None else "failed",
            "eye_detector": "loaded" if detector.eye_cascade is not None else "failed",
            "smile_detector": "loaded" if detector.smile_cascade is not None else "failed",
            "profile_detector": "loaded" if detector.profile_cascade is not None else "failed"
        },
        "capabilities": [
            "ultra_facial_symmetry_analysis",
            "advanced_texture_analysis_with_lbp_glcm_gabor",
            "comprehensive_eye_iris_analysis",
            "advanced_skin_authenticity_detection",
            "micro_expression_analysis",
            "lighting_consistency_verification",
            "frequency_domain_analysis",
            "biometric_consistency_analysis",
            "multi_scale_face_detection",
            "ultra_precision_ensemble_scoring"
        ]
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

    logger.info(f"Starting enhanced analysis for: {file_path}")
    
    try:
        result = convert_numpy_types(detector.analyze_image(file_path))
        logger.info(f"Ultra-precision analysis complete: {result['prediction']} (confidence: {result.get('confidence', 0):.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ultra-precision analysis: {e}")
        return jsonify({
            "error": "Ultra-precision analysis failed",
            "details": str(e),
            "prediction": "ERROR"
        }), 500
    
def convert_numpy_types(obj):
    """
    Recursively convert numpy data types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)