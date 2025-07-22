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

class EnhancedDeepfakeDetector:
    """Enhanced deepfake detector with advanced computer vision and ML techniques"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self._load_detectors()
    
    def _load_detectors(self):
        """Load OpenCV detectors"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            logger.info("All detectors loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load detectors: {e}")
    
    def detect_facial_features(self, image_path):
        """Enhanced facial feature detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            facial_analysis = {
                'face_count': len(faces),
                'faces': faces.tolist(),
                'face_details': []
            }
            
            for (x, y, w, h) in faces:
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = image[y:y+h, x:x+w]
                
                # Detect eyes within face
                eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3)
                
                # Detect smile
                smiles = self.smile_cascade.detectMultiScale(face_roi_gray, 1.8, 20)
                
                # Calculate face symmetry
                symmetry_score = self._calculate_face_symmetry(face_roi_gray)
                
                # Analyze skin texture
                texture_analysis = self._analyze_skin_texture(face_roi_color)
                
                # Eye analysis
                eye_analysis = self._analyze_eyes(face_roi_gray, eyes)
                
                face_detail = {
                    'position': [int(x), int(y), int(w), int(h)],
                    'eye_count': len(eyes),
                    'eyes': eyes.tolist(),
                    'smile_count': len(smiles),
                    'symmetry_score': float(symmetry_score),
                    'texture_analysis': texture_analysis,
                    'eye_analysis': eye_analysis,
                    'face_area': int(w * h),
                    'aspect_ratio': float(w / h) if h > 0 else 0
                }
                
                facial_analysis['face_details'].append(face_detail)
            
            return facial_analysis
            
        except Exception as e:
            logger.error(f"Facial feature detection error: {e}")
            return {'error': str(e)}
    
    def _calculate_face_symmetry(self, face_gray):
        """Calculate facial symmetry score"""
        try:
            h, w = face_gray.shape
            left_half = face_gray[:, :w//2]
            right_half = cv2.flip(face_gray[:, w//2:], 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate correlation
            correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            return max(0, correlation)
            
        except Exception as e:
            logger.error(f"Symmetry calculation error: {e}")
            return 0.0
    
    def _analyze_skin_texture(self, face_roi):
        """Analyze skin texture for unnatural smoothness"""
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern for texture analysis
            lbp = feature.local_binary_pattern(gray_face, 8, 1, method='uniform')
            lbp_hist = np.histogram(lbp.ravel(), bins=10)[0]
            lbp_uniformity = np.std(lbp_hist)
            
            # Gradient analysis
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            gradient_std = np.std(gradient_magnitude)
            
            # Frequency domain analysis
            f_transform = np.fft.fft2(gray_face)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            high_freq_energy = np.mean(magnitude_spectrum[gray_face.shape[0]//4:3*gray_face.shape[0]//4, 
                                                        gray_face.shape[1]//4:3*gray_face.shape[1]//4])
            
            return {
                'lbp_uniformity': float(lbp_uniformity),
                'gradient_mean': float(gradient_mean),
                'gradient_std': float(gradient_std),
                'high_freq_energy': float(high_freq_energy),
                'is_overly_smooth': gradient_std < 10 and lbp_uniformity < 50
            }
            
        except Exception as e:
            logger.error(f"Texture analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_eyes(self, face_gray, eyes):
        """Analyze eye characteristics"""
        try:
            if len(eyes) < 2:
                return {'error': 'Insufficient eyes detected', 'eye_count': len(eyes)}
            
            eye_analysis = {
                'eye_count': len(eyes),
                'eye_symmetry': 0.0,
                'eye_spacing_ratio': 0.0,
                'eye_size_consistency': 0.0
            }
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes_sorted[0], eyes_sorted[1]
                
                # Eye spacing analysis
                eye_distance = abs(left_eye[0] - right_eye[0])
                face_width = face_gray.shape[1]
                eye_spacing_ratio = eye_distance / face_width if face_width > 0 else 0
                
                # Eye size consistency
                left_area = left_eye[2] * left_eye[3]
                right_area = right_eye[2] * right_eye[3]
                size_ratio = min(left_area, right_area) / max(left_area, right_area) if max(left_area, right_area) > 0 else 0
                
                eye_analysis.update({
                    'eye_spacing_ratio': float(eye_spacing_ratio),
                    'eye_size_consistency': float(size_ratio),
                    'left_eye_area': int(left_area),
                    'right_eye_area': int(right_area)
                })
            
            return eye_analysis
            
        except Exception as e:
            logger.error(f"Eye analysis error: {e}")
            return {'error': str(e)}
    
    def advanced_quality_analysis(self, image_path):
        """Enhanced image quality analysis"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic quality metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Advanced blur detection using multiple methods
            blur_scores = {
                'laplacian': float(laplacian_var),
                'sobel': float(np.var(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))),
                'brenner': self._brenner_focus_measure(gray),
                'tenengrad': self._tenengrad_focus_measure(gray)
            }
            
            # Noise analysis
            noise_level = self._estimate_noise_level(gray)
            
            # Compression artifact detection
            compression_analysis = self._advanced_compression_detection(image)
            
            # Color consistency analysis
            color_analysis = self._analyze_color_consistency(image)
            
            # Edge coherence analysis
            edge_analysis = self._analyze_edge_coherence(gray)
            
            return {
                'blur_scores': blur_scores,
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(std_brightness),
                'noise_level': noise_level,
                'compression_analysis': compression_analysis,
                'color_analysis': color_analysis,
                'edge_analysis': edge_analysis,
                'overall_quality_score': self._calculate_quality_score(blur_scores, noise_level, compression_analysis)
            }
            
        except Exception as e:
            logger.error(f"Advanced quality analysis error: {e}")
            return {'error': str(e)}
    
    def _brenner_focus_measure(self, gray):
        """Brenner focus measure for blur detection"""
        try:
            brenner = np.sum((gray[:-2, :] - gray[2:, :])**2)
            return float(brenner)
        except:
            return 0.0
    
    def _tenengrad_focus_measure(self, gray):
        """Tenengrad focus measure"""
        try:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.sum(gx**2 + gy**2)
            return float(tenengrad)
        except:
            return 0.0
    
    def _estimate_noise_level(self, gray):
        """Estimate noise level in the image"""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_estimate = np.var(laplacian)
            return float(noise_estimate)
        except:
            return 0.0
    
    def _advanced_compression_detection(self, image):
        """Advanced JPEG compression artifact detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # DCT-based artifact detection
            block_size = 8
            artifact_scores = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    if block.shape == (block_size, block_size):
                        dct_block = cv2.dct(block)
                        # Analyze high-frequency components
                        high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                        artifact_scores.append(high_freq)
            
            if artifact_scores:
                mean_artifact = np.mean(artifact_scores)
                std_artifact = np.std(artifact_scores)
                
                # Blocking artifact detection
                blocking_score = self._detect_blocking_artifacts(gray)
                
                return {
                    'mean_artifact_score': float(mean_artifact),
                    'artifact_std': float(std_artifact),
                    'blocking_score': blocking_score,
                    'has_significant_artifacts': mean_artifact > 1000 or blocking_score > 0.3
                }
            
            return {'error': 'Could not analyze compression artifacts'}
            
        except Exception as e:
            logger.error(f"Compression detection error: {e}")
            return {'error': str(e)}
    
    def _detect_blocking_artifacts(self, gray):
        """Detect JPEG blocking artifacts"""
        try:
            # Detect regular patterns that indicate blocking
            h, w = gray.shape
            block_boundaries_h = []
            block_boundaries_v = []
            
            # Check for horizontal blocking patterns
            for i in range(8, h-8, 8):
                diff = np.mean(np.abs(gray[i-1, :] - gray[i, :]))
                block_boundaries_h.append(diff)
            
            # Check for vertical blocking patterns
            for j in range(8, w-8, 8):
                diff = np.mean(np.abs(gray[:, j-1] - gray[:, j]))
                block_boundaries_v.append(diff)
            
            if block_boundaries_h and block_boundaries_v:
                blocking_score = (np.mean(block_boundaries_h) + np.mean(block_boundaries_v)) / 2
                return float(blocking_score)
            
            return 0.0
            
        except:
            return 0.0
    
    def _analyze_color_consistency(self, image):
        """Analyze color consistency across the image"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
            # Color temperature consistency
            b, g, r = cv2.split(image)
            color_temp_ratio = np.mean(b) / (np.mean(r) + 1e-6)
            
            return {
                'hue_std': float(h_std),
                'saturation_std': float(s_std),
                'value_std': float(v_std),
                'color_temp_ratio': float(color_temp_ratio),
                'color_inconsistency': h_std > 30 or color_temp_ratio > 1.5 or color_temp_ratio < 0.5
            }
            
        except Exception as e:
            logger.error(f"Color analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_edge_coherence(self, gray):
        """Analyze edge coherence and consistency"""
        try:
            # Multiple edge detection methods
            canny = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Edge density
            edge_density = np.sum(canny > 0) / (canny.shape[0] * canny.shape[1])
            
            # Edge strength distribution
            edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_mean = np.mean(edge_strength)
            edge_std = np.std(edge_strength)
            
            # Edge direction consistency
            edge_direction = np.arctan2(sobel_y, sobel_x)
            direction_consistency = np.std(edge_direction[edge_strength > np.percentile(edge_strength, 75)])
            
            return {
                'edge_density': float(edge_density),
                'edge_strength_mean': float(edge_mean),
                'edge_strength_std': float(edge_std),
                'direction_consistency': float(direction_consistency),
                'edge_anomaly': edge_density < 0.02 or edge_density > 0.4 or direction_consistency > 2.0
            }
            
        except Exception as e:
            logger.error(f"Edge analysis error: {e}")
            return {'error': str(e)}
    
    def _calculate_quality_score(self, blur_scores, noise_level, compression_analysis):
        """Calculate overall quality score"""
        try:
            # Normalize and combine different quality metrics
            blur_score = np.mean(list(blur_scores.values()))
            normalized_blur = min(1.0, blur_score / 1000)
            
            normalized_noise = min(1.0, noise_level / 10000)
            
            compression_score = compression_analysis.get('mean_artifact_score', 0)
            normalized_compression = min(1.0, compression_score / 2000)
            
            # Weighted combination
            quality_score = (0.4 * normalized_blur + 0.3 * (1 - normalized_noise) + 0.3 * (1 - normalized_compression))
            return float(max(0, min(1, quality_score)))
            
        except:
            return 0.5
    
    def ensemble_deepfake_detection(self, image_path, facial_analysis, quality_analysis):
        """Enhanced ensemble method for deepfake detection"""
        try:
            suspicion_factors = []
            confidence_factors = []
            detailed_reasons = []
            
            # Facial analysis scoring
            if facial_analysis.get('face_count', 0) == 0:
                suspicion_factors.append(0.4)
                detailed_reasons.append("No faces detected in image")
            elif facial_analysis.get('face_count', 0) > 3:
                suspicion_factors.append(0.2)
                detailed_reasons.append("Unusual number of faces detected")
            
            # Analyze each face
            for i, face_detail in enumerate(facial_analysis.get('face_details', [])):
                face_suspicion = 0.0
                
                # Symmetry analysis
                symmetry = face_detail.get('symmetry_score', 0)
                if symmetry > 0.95:  # Too perfect symmetry
                    face_suspicion += 0.15
                    detailed_reasons.append(f"Face {i+1}: Unnaturally high symmetry ({symmetry:.2f})")
                elif symmetry < 0.3:  # Too asymmetric
                    face_suspicion += 0.1
                    detailed_reasons.append(f"Face {i+1}: Unusual asymmetry")
                
                # Eye analysis
                eye_analysis = face_detail.get('eye_analysis', {})
                if eye_analysis.get('eye_count', 0) != 2:
                    face_suspicion += 0.2
                    detailed_reasons.append(f"Face {i+1}: Abnormal eye detection")
                
                eye_consistency = eye_analysis.get('eye_size_consistency', 1.0)
                if eye_consistency < 0.7:
                    face_suspicion += 0.15
                    detailed_reasons.append(f"Face {i+1}: Inconsistent eye sizes")
                
                # Texture analysis
                texture = face_detail.get('texture_analysis', {})
                if texture.get('is_overly_smooth', False):
                    face_suspicion += 0.25
                    detailed_reasons.append(f"Face {i+1}: Unnaturally smooth skin texture")
                
                if texture.get('gradient_std', 0) < 5:
                    face_suspicion += 0.1
                    detailed_reasons.append(f"Face {i+1}: Low texture variation")
                
                suspicion_factors.append(face_suspicion)
            
            # Quality analysis scoring
            quality_score = quality_analysis.get('overall_quality_score', 0.5)
            if quality_score < 0.3:
                suspicion_factors.append(0.2)
                detailed_reasons.append("Poor overall image quality")
            
            # Blur analysis
            blur_scores = quality_analysis.get('blur_scores', {})
            avg_blur = np.mean(list(blur_scores.values())) if blur_scores else 0
            if avg_blur < 100:
                suspicion_factors.append(0.15)
                detailed_reasons.append("Image appears significantly blurred")
            
            # Compression artifacts
            compression = quality_analysis.get('compression_analysis', {})
            if compression.get('has_significant_artifacts', False):
                suspicion_factors.append(0.2)
                detailed_reasons.append("Significant compression artifacts detected")
            
            # Color consistency
            color_analysis = quality_analysis.get('color_analysis', {})
            if color_analysis.get('color_inconsistency', False):
                suspicion_factors.append(0.15)
                detailed_reasons.append("Color inconsistencies detected")
            
            # Edge analysis
            edge_analysis = quality_analysis.get('edge_analysis', {})
            if edge_analysis.get('edge_anomaly', False):
                suspicion_factors.append(0.1)
                detailed_reasons.append("Edge pattern anomalies detected")
            
            # Calculate final scores
            total_suspicion = sum(suspicion_factors)
            max_suspicion = min(1.0, total_suspicion)
            
            # Determine prediction with more nuanced thresholds
            if max_suspicion >= 0.7:
                prediction = "FAKE"
                confidence = min(0.95, 0.6 + max_suspicion * 0.35)
                risk_level = "HIGH"
            elif max_suspicion >= 0.4:
                prediction = "SUSPICIOUS"
                confidence = 0.65 + (max_suspicion - 0.4) * 0.2
                risk_level = "MEDIUM"
            elif max_suspicion >= 0.2:
                prediction = "LIKELY_REAL"
                confidence = 0.7 + (0.4 - max_suspicion) * 0.2
                risk_level = "LOW"
            else:
                prediction = "REAL"
                confidence = min(0.95, 0.8 + (0.2 - max_suspicion) * 0.75)
                risk_level = "LOW"
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'suspicion_score': max_suspicion,
                'risk_level': risk_level,
                'detailed_reasons': detailed_reasons,
                'suspicion_factors': suspicion_factors,
                'analysis_depth': 'enhanced_ensemble'
            }
            
        except Exception as e:
            logger.error(f"Ensemble detection error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'analysis_depth': 'enhanced_ensemble'
            }
    
    def analyze_image(self, image_path):
        """Main enhanced analysis function"""
        try:
            logger.info(f"Starting enhanced analysis for: {image_path}")
            
            # Enhanced facial feature detection
            facial_analysis = self.detect_facial_features(image_path)
            
            # Advanced quality analysis
            quality_analysis = self.advanced_quality_analysis(image_path)
            
            # Ensemble deepfake detection
            detection_result = self.ensemble_deepfake_detection(image_path, facial_analysis, quality_analysis)
            
            # Compile comprehensive result
            result = {
                'prediction': detection_result['prediction'],
                'confidence': detection_result['confidence'],
                'risk_level': detection_result['risk_level'],
                'suspicion_score': detection_result['suspicion_score'],
                'has_faces': facial_analysis.get('face_count', 0) > 0,
                'face_count': facial_analysis.get('face_count', 0),
                'faces': facial_analysis.get('faces', []),
                'facial_analysis': facial_analysis,
                'quality_analysis': quality_analysis,
                'detection_reasons': detection_result.get('detailed_reasons', []),
                'model_used': 'enhanced_ensemble_cv',
                'analysis_depth': 'comprehensive'
            }
            
            # Generate risk factors summary
            risk_factors = []
            if not result['has_faces']:
                risk_factors.append("No faces detected")
            if quality_analysis.get('overall_quality_score', 0.5) < 0.4:
                risk_factors.append("Poor image quality")
            if detection_result['confidence'] < 0.7:
                risk_factors.append("Low confidence prediction")
            if len(detection_result.get('detailed_reasons', [])) > 3:
                risk_factors.append("Multiple anomalies detected")
            
            result['risk_factors'] = risk_factors
            
            logger.info(f"Enhanced analysis complete: {result['prediction']} (confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced analysis error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'model_used': 'enhanced_ensemble_cv',
                'analysis_depth': 'error'
            }

# Initialize the enhanced detection service
detector = EnhancedDeepfakeDetector()

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
        "message": "SheGuard Enhanced Backend is running!",
        "model_status": "enhanced_ensemble_cv",
        "version": "4.0-enhanced",
        "features": [
            "Advanced facial feature detection",
            "Multi-method blur detection",
            "Texture analysis with LBP",
            "Compression artifact detection",
            "Color consistency analysis",
            "Edge coherence analysis",
            "Ensemble prediction system"
        ]
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_type": "enhanced_ensemble_cv",
        "detectors": {
            "face_detector": "loaded" if detector.face_cascade is not None else "failed",
            "eye_detector": "loaded" if detector.eye_cascade is not None else "failed",
            "smile_detector": "loaded" if detector.smile_cascade is not None else "failed"
        },
        "capabilities": [
            "facial_symmetry_analysis",
            "skin_texture_analysis", 
            "eye_consistency_check",
            "advanced_blur_detection",
            "compression_artifact_detection",
            "color_consistency_analysis",
            "edge_coherence_analysis"
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
        logger.info(f"Enhanced analysis complete: {result['prediction']} (confidence: {result.get('confidence', 0):.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        return jsonify({
            "error": "Enhanced analysis failed",
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