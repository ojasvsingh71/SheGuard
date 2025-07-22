import cv2
import numpy as np
from scipy import ndimage, signal, stats
from skimage import feature, measure, filters, segmentation, morphology, restoration
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math
import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class UltraPrecisionDeepfakeDetector:
    """Ultra-precision deepfake detector with advanced ML and CV techniques"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self.profile_cascade = None
        self._load_detectors()
        
        # Advanced detection thresholds (fine-tuned)
        self.thresholds = {
            'symmetry_perfect': 0.98,
            'symmetry_suspicious': 0.95,
            'texture_smoothness': 8.0,
            'gradient_threshold': 15.0,
            'frequency_anomaly': 0.3,
            'eye_consistency': 0.75,
            'skin_uniformity': 0.85,
            'compression_artifact': 1500,
            'edge_coherence': 0.25,
            'color_temperature_variance': 0.4,
            'micro_expression_threshold': 0.1,
            'lighting_consistency': 0.3
        }
    
    def _load_detectors(self):
        """Load all available OpenCV detectors"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            logger.info("All advanced detectors loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load detectors: {e}")
    
    def ultra_facial_analysis(self, image_path):
        """Ultra-comprehensive facial analysis with advanced techniques"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale face detection
            faces_frontal = self.face_cascade.detectMultiScale(gray, 1.05, 6, minSize=(50, 50))
            faces_profile = self.profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            all_faces = np.vstack([faces_frontal, faces_profile]) if len(faces_profile) > 0 else faces_frontal
            
            facial_analysis = {
                'face_count': len(all_faces),
                'frontal_faces': len(faces_frontal),
                'profile_faces': len(faces_profile),
                'faces': all_faces.tolist(),
                'face_details': [],
                'global_facial_metrics': {}
            }
            
            if len(all_faces) == 0:
                return facial_analysis
            
            # Global facial analysis
            facial_analysis['global_facial_metrics'] = self._analyze_global_facial_patterns(gray, all_faces)
            
            for i, (x, y, w, h) in enumerate(all_faces):
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = image[y:y+h, x:x+w]
                
                # Comprehensive face analysis
                face_detail = {
                    'face_id': i,
                    'position': [int(x), int(y), int(w), int(h)],
                    'face_area': int(w * h),
                    'aspect_ratio': float(w / h) if h > 0 else 0,
                    
                    # Advanced geometric analysis
                    'geometric_analysis': self._advanced_geometric_analysis(face_roi_gray),
                    
                    # Multi-method symmetry analysis
                    'symmetry_analysis': self._multi_method_symmetry_analysis(face_roi_gray),
                    
                    # Advanced texture analysis
                    'texture_analysis': self._ultra_texture_analysis(face_roi_color),
                    
                    # Comprehensive eye analysis
                    'eye_analysis': self._comprehensive_eye_analysis(face_roi_gray, face_roi_color),
                    
                    # Skin analysis
                    'skin_analysis': self._advanced_skin_analysis(face_roi_color),
                    
                    # Micro-expression analysis
                    'micro_expression_analysis': self._micro_expression_analysis(face_roi_gray),
                    
                    # Lighting consistency
                    'lighting_analysis': self._lighting_consistency_analysis(face_roi_color),
                    
                    # Frequency domain analysis
                    'frequency_analysis': self._frequency_domain_analysis(face_roi_gray),
                    
                    # Biometric consistency
                    'biometric_analysis': self._biometric_consistency_analysis(face_roi_gray)
                }
                
                facial_analysis['face_details'].append(face_detail)
            
            return facial_analysis
            
        except Exception as e:
            logger.error(f"Ultra facial analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_global_facial_patterns(self, gray, faces):
        """Analyze global patterns across all detected faces"""
        try:
            if len(faces) == 0:
                return {}
            
            # Face size consistency
            face_areas = [w * h for (x, y, w, h) in faces]
            size_variance = np.var(face_areas) / (np.mean(face_areas) + 1e-6)
            
            # Face positioning patterns
            face_centers = [(x + w/2, y + h/2) for (x, y, w, h) in faces]
            if len(face_centers) > 1:
                distances = []
                for i in range(len(face_centers)):
                    for j in range(i+1, len(face_centers)):
                        dist = np.sqrt((face_centers[i][0] - face_centers[j][0])**2 + 
                                     (face_centers[i][1] - face_centers[j][1])**2)
                        distances.append(dist)
                distance_variance = np.var(distances) if distances else 0
            else:
                distance_variance = 0
            
            return {
                'face_count': len(faces),
                'size_variance': float(size_variance),
                'distance_variance': float(distance_variance),
                'average_face_area': float(np.mean(face_areas)),
                'size_consistency_score': float(1.0 / (1.0 + size_variance))
            }
            
        except Exception as e:
            logger.error(f"Global facial pattern analysis error: {e}")
            return {}
    
    def _advanced_geometric_analysis(self, face_gray):
        """Advanced geometric analysis of facial features"""
        try:
            h, w = face_gray.shape
            
            # Golden ratio analysis
            golden_ratio = 1.618
            face_ratio = w / h if h > 0 else 0
            golden_ratio_deviation = abs(face_ratio - golden_ratio) / golden_ratio
            
            # Facial thirds analysis
            upper_third = face_gray[:h//3, :]
            middle_third = face_gray[h//3:2*h//3, :]
            lower_third = face_gray[2*h//3:, :]
            
            third_intensities = [
                np.mean(upper_third),
                np.mean(middle_third),
                np.mean(lower_third)
            ]
            third_variance = np.var(third_intensities)
            
            # Contour analysis
            edges = cv2.Canny(face_gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                contour_perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * contour_area / (contour_perimeter**2) if contour_perimeter > 0 else 0
                
                # Convex hull analysis
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
            else:
                circularity = 0
                solidity = 0
            
            return {
                'golden_ratio_deviation': float(golden_ratio_deviation),
                'face_ratio': float(face_ratio),
                'third_variance': float(third_variance),
                'circularity': float(circularity),
                'solidity': float(solidity),
                'geometric_anomaly': golden_ratio_deviation > 0.3 or third_variance > 500
            }
            
        except Exception as e:
            logger.error(f"Geometric analysis error: {e}")
            return {'error': str(e)}
    
    def _multi_method_symmetry_analysis(self, face_gray):
        """Multi-method symmetry analysis with advanced techniques"""
        try:
            h, w = face_gray.shape
            
            # Method 1: Direct pixel comparison
            left_half = face_gray[:, :w//2]
            right_half = cv2.flip(face_gray[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            pixel_correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Method 2: Feature-based symmetry
            orb = cv2.ORB_create(nfeatures=100)
            kp1, des1 = orb.detectAndCompute(left_half, None)
            kp2, des2 = orb.detectAndCompute(right_half, None)
            
            feature_symmetry = 0.0
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                feature_symmetry = len(matches) / max(len(des1), len(des2))
            
            # Method 3: Gradient-based symmetry
            grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            
            left_grad_x = grad_x[:, :w//2]
            right_grad_x = cv2.flip(grad_x[:, w//2:], 1)
            right_grad_x = -right_grad_x  # Flip gradient direction
            
            min_width = min(left_grad_x.shape[1], right_grad_x.shape[1])
            left_grad_x = left_grad_x[:, :min_width]
            right_grad_x = right_grad_x[:, :min_width]
            
            gradient_correlation = np.corrcoef(left_grad_x.flatten(), right_grad_x.flatten())[0, 1]
            gradient_correlation = max(0, gradient_correlation) if not np.isnan(gradient_correlation) else 0
            
            # Method 4: Frequency domain symmetry
            f_transform = np.fft.fft2(face_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            left_freq = magnitude[:, :w//2]
            right_freq = cv2.flip(magnitude[:, w//2:], 1)
            min_width = min(left_freq.shape[1], right_freq.shape[1])
            left_freq = left_freq[:, :min_width]
            right_freq = right_freq[:, :min_width]
            
            freq_correlation = np.corrcoef(left_freq.flatten(), right_freq.flatten())[0, 1]
            freq_correlation = max(0, freq_correlation) if not np.isnan(freq_correlation) else 0
            
            # Combined symmetry score
            symmetry_scores = [pixel_correlation, feature_symmetry, gradient_correlation, freq_correlation]
            combined_symmetry = np.mean([s for s in symmetry_scores if not np.isnan(s)])
            
            return {
                'pixel_symmetry': float(pixel_correlation),
                'feature_symmetry': float(feature_symmetry),
                'gradient_symmetry': float(gradient_correlation),
                'frequency_symmetry': float(freq_correlation),
                'combined_symmetry': float(combined_symmetry),
                'symmetry_variance': float(np.var(symmetry_scores)),
                'is_unnaturally_symmetric': combined_symmetry > self.thresholds['symmetry_suspicious'],
                'is_perfectly_symmetric': combined_symmetry > self.thresholds['symmetry_perfect']
            }
            
        except Exception as e:
            logger.error(f"Multi-method symmetry analysis error: {e}")
            return {'error': str(e)}
    
    def _ultra_texture_analysis(self, face_roi_color):
        """Ultra-comprehensive texture analysis"""
        try:
            gray_face = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY)
            
            # 1. Local Binary Pattern analysis (multiple radii)
            lbp_results = {}
            for radius in [1, 2, 3]:
                n_points = 8 * radius
                lbp = feature.local_binary_pattern(gray_face, n_points, radius, method='uniform')
                lbp_hist = np.histogram(lbp.ravel(), bins=n_points + 2)[0]
                lbp_results[f'lbp_r{radius}_uniformity'] = float(np.std(lbp_hist))
                lbp_results[f'lbp_r{radius}_entropy'] = float(stats.entropy(lbp_hist + 1e-10))
            
            # 2. Gray-Level Co-occurrence Matrix (GLCM)
            glcm = feature.graycomatrix(gray_face.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            glcm_props = {
                'contrast': float(np.mean(feature.graycoprops(glcm, 'contrast'))),
                'dissimilarity': float(np.mean(feature.graycoprops(glcm, 'dissimilarity'))),
                'homogeneity': float(np.mean(feature.graycoprops(glcm, 'homogeneity'))),
                'energy': float(np.mean(feature.graycoprops(glcm, 'energy'))),
                'correlation': float(np.mean(feature.graycoprops(glcm, 'correlation')))
            }
            
            # 3. Wavelet texture analysis
            from scipy import ndimage
            
            # Gabor filters at different orientations and frequencies
            gabor_responses = []
            for theta in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3, 0.5]:
                    real, _ = filters.gabor(gray_face, frequency=frequency, theta=np.deg2rad(theta))
                    gabor_responses.append(np.var(real))
            
            gabor_mean = float(np.mean(gabor_responses))
            gabor_std = float(np.std(gabor_responses))
            
            # 4. Fractal dimension analysis
            def box_count(image, min_box_size=1, max_box_size=None):
                if max_box_size is None:
                    max_box_size = min(image.shape) // 4
                
                sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10, dtype=int)
                counts = []
                
                for size in sizes:
                    if size >= min(image.shape):
                        continue
                    
                    # Divide image into boxes
                    h, w = image.shape
                    n_boxes_h = h // size
                    n_boxes_w = w // size
                    
                    count = 0
                    for i in range(n_boxes_h):
                        for j in range(n_boxes_w):
                            box = image[i*size:(i+1)*size, j*size:(j+1)*size]
                            if np.var(box) > 10:  # Box contains texture
                                count += 1
                    counts.append(count)
                
                if len(counts) > 1 and len(sizes) > 1:
                    coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
                    return -coeffs[0]
                return 1.5
            
            fractal_dimension = box_count(gray_face)
            
            # 5. Statistical texture measures
            mean_intensity = float(np.mean(gray_face))
            std_intensity = float(np.std(gray_face))
            skewness = float(stats.skew(gray_face.flatten()))
            kurtosis = float(stats.kurtosis(gray_face.flatten()))
            
            # 6. Edge-based texture analysis
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            
            # Gradient magnitude and direction
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            grad_mag_mean = float(np.mean(gradient_magnitude))
            grad_mag_std = float(np.std(gradient_magnitude))
            grad_dir_std = float(np.std(gradient_direction))
            
            # 7. Smoothness detection
            laplacian_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
            
            # Combine all texture features
            texture_analysis = {
                **lbp_results,
                **glcm_props,
                'gabor_mean': gabor_mean,
                'gabor_std': gabor_std,
                'fractal_dimension': float(fractal_dimension),
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'edge_density': edge_density,
                'gradient_magnitude_mean': grad_mag_mean,
                'gradient_magnitude_std': grad_mag_std,
                'gradient_direction_std': grad_dir_std,
                'laplacian_variance': laplacian_var,
                
                # Derived indicators
                'is_overly_smooth': grad_mag_std < self.thresholds['texture_smoothness'],
                'has_unnatural_uniformity': std_intensity < 10 and glcm_props['homogeneity'] > 0.8,
                'texture_complexity_score': float(gabor_std * fractal_dimension * grad_mag_std / 1000),
                'artificial_texture_probability': float(1.0 / (1.0 + np.exp(-(0.8 - glcm_props['homogeneity']) * 10)))
            }
            
            return texture_analysis
            
        except Exception as e:
            logger.error(f"Ultra texture analysis error: {e}")
            return {'error': str(e)}
    
    def _comprehensive_eye_analysis(self, face_gray, face_color):
        """Comprehensive eye analysis with advanced techniques"""
        try:
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            
            eye_analysis = {
                'eye_count': len(eyes),
                'eyes_detected': eyes.tolist(),
                'eye_details': []
            }
            
            if len(eyes) == 0:
                return eye_analysis
            
            # Analyze each eye
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                eye_roi_gray = face_gray[ey:ey+eh, ex:ex+ew]
                eye_roi_color = face_color[ey:ey+eh, ex:ex+ew]
                
                # Pupil detection using HoughCircles
                circles = cv2.HoughCircles(eye_roi_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=5, maxRadius=25)
                
                pupil_detected = circles is not None and len(circles[0]) > 0
                pupil_info = {}
                
                if pupil_detected:
                    circle = circles[0][0]
                    pupil_info = {
                        'center': [float(circle[0]), float(circle[1])],
                        'radius': float(circle[2]),
                        'area': float(np.pi * circle[2]**2)
                    }
                
                # Eye texture analysis
                eye_texture = self._analyze_eye_texture(eye_roi_gray)
                
                # Iris pattern analysis
                iris_analysis = self._analyze_iris_patterns(eye_roi_gray)
                
                # Reflection analysis
                reflection_analysis = self._analyze_eye_reflections(eye_roi_color)
                
                eye_detail = {
                    'eye_id': i,
                    'position': [int(ex), int(ey), int(ew), int(eh)],
                    'area': int(ew * eh),
                    'aspect_ratio': float(ew / eh) if eh > 0 else 0,
                    'pupil_detected': pupil_detected,
                    'pupil_info': pupil_info,
                    'texture_analysis': eye_texture,
                    'iris_analysis': iris_analysis,
                    'reflection_analysis': reflection_analysis
                }
                
                eye_analysis['eye_details'].append(eye_detail)
            
            # Cross-eye consistency analysis
            if len(eyes) >= 2:
                eye_analysis['consistency_analysis'] = self._analyze_eye_consistency(eye_analysis['eye_details'])
            
            return eye_analysis
            
        except Exception as e:
            logger.error(f"Comprehensive eye analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_eye_texture(self, eye_gray):
        """Analyze eye texture patterns"""
        try:
            # Local Binary Pattern for eye texture
            lbp = feature.local_binary_pattern(eye_gray, 8, 1, method='uniform')
            lbp_hist = np.histogram(lbp.ravel(), bins=10)[0]
            lbp_uniformity = float(np.std(lbp_hist))
            
            # Gradient analysis
            grad_x = cv2.Sobel(eye_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(eye_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return {
                'lbp_uniformity': lbp_uniformity,
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'texture_complexity': float(lbp_uniformity * np.std(gradient_magnitude))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_iris_patterns(self, eye_gray):
        """Analyze iris patterns for authenticity"""
        try:
            # Apply circular mask to focus on iris region
            h, w = eye_gray.shape
            center = (w//2, h//2)
            radius = min(w, h) // 3
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            iris_region = cv2.bitwise_and(eye_gray, mask)
            
            # Radial pattern analysis
            angles = np.linspace(0, 2*np.pi, 36)
            radial_profiles = []
            
            for angle in angles:
                x_coords = center[0] + np.arange(0, radius) * np.cos(angle)
                y_coords = center[1] + np.arange(0, radius) * np.sin(angle)
                
                x_coords = np.clip(x_coords.astype(int), 0, w-1)
                y_coords = np.clip(y_coords.astype(int), 0, h-1)
                
                profile = iris_region[y_coords, x_coords]
                radial_profiles.append(np.var(profile))
            
            radial_variance = float(np.var(radial_profiles))
            radial_mean = float(np.mean(radial_profiles))
            
            return {
                'radial_variance': radial_variance,
                'radial_mean': radial_mean,
                'pattern_complexity': float(radial_variance / (radial_mean + 1e-6)),
                'has_natural_patterns': radial_variance > 50
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_eye_reflections(self, eye_color):
        """Analyze eye reflections for lighting consistency"""
        try:
            # Convert to HSV for better reflection detection
            hsv = cv2.cvtColor(eye_color, cv2.COLOR_BGR2HSV)
            
            # Detect bright spots (reflections)
            _, bright_mask = cv2.threshold(hsv[:,:,2], 200, 255, cv2.THRESH_BINARY)
            
            # Find reflection contours
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            reflection_count = len(contours)
            reflection_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 5]
            
            return {
                'reflection_count': reflection_count,
                'total_reflection_area': float(sum(reflection_areas)),
                'average_reflection_size': float(np.mean(reflection_areas)) if reflection_areas else 0,
                'has_natural_reflections': 1 <= reflection_count <= 3
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_eye_consistency(self, eye_details):
        """Analyze consistency between multiple eyes"""
        try:
            if len(eye_details) < 2:
                return {'error': 'Insufficient eyes for consistency analysis'}
            
            # Size consistency
            areas = [eye['area'] for eye in eye_details]
            size_variance = float(np.var(areas) / (np.mean(areas) + 1e-6))
            
            # Texture consistency
            texture_complexities = [eye['texture_analysis'].get('texture_complexity', 0) for eye in eye_details]
            texture_variance = float(np.var(texture_complexities))
            
            # Reflection consistency
            reflection_counts = [eye['reflection_analysis'].get('reflection_count', 0) for eye in eye_details]
            reflection_consistency = float(np.std(reflection_counts))
            
            return {
                'size_consistency': float(1.0 / (1.0 + size_variance)),
                'texture_consistency': float(1.0 / (1.0 + texture_variance)),
                'reflection_consistency': float(1.0 / (1.0 + reflection_consistency)),
                'overall_consistency': float(1.0 / (1.0 + size_variance + texture_variance + reflection_consistency))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _advanced_skin_analysis(self, face_color):
        """Advanced skin analysis for authenticity detection"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_color, cv2.COLOR_BGR2LAB)
            yuv = cv2.cvtColor(face_color, cv2.COLOR_BGR2YUV)
            
            # Skin segmentation using multiple methods
            skin_mask1 = self._skin_segmentation_hsv(hsv)
            skin_mask2 = self._skin_segmentation_yuv(yuv)
            combined_skin_mask = cv2.bitwise_and(skin_mask1, skin_mask2)
            
            # Extract skin regions
            skin_pixels_bgr = face_color[combined_skin_mask > 0]
            skin_pixels_hsv = hsv[combined_skin_mask > 0]
            skin_pixels_lab = lab[combined_skin_mask > 0]
            
            if len(skin_pixels_bgr) == 0:
                return {'error': 'No skin pixels detected'}
            
            # Color distribution analysis
            color_analysis = {
                'mean_hue': float(np.mean(skin_pixels_hsv[:, 0])),
                'std_hue': float(np.std(skin_pixels_hsv[:, 0])),
                'mean_saturation': float(np.mean(skin_pixels_hsv[:, 1])),
                'std_saturation': float(np.std(skin_pixels_hsv[:, 1])),
                'mean_value': float(np.mean(skin_pixels_hsv[:, 2])),
                'std_value': float(np.std(skin_pixels_hsv[:, 2]))
            }
            
            # Skin texture uniformity
            skin_region = cv2.bitwise_and(face_color, face_color, mask=combined_skin_mask)
            gray_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
            
            # Local variance analysis
            kernel = np.ones((5,5), np.float32) / 25
            local_mean = cv2.filter2D(gray_skin.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_skin.astype(np.float32) - local_mean)**2, -1, kernel)
            
            uniformity_score = float(np.mean(local_variance[combined_skin_mask > 0]))
            
            # Skin color temperature analysis
            b, g, r = cv2.split(skin_pixels_bgr)
            color_temperature = float(np.mean(b) / (np.mean(r) + 1e-6))
            
            # Subsurface scattering simulation
            red_channel = skin_pixels_bgr[:, 2]
            green_channel = skin_pixels_bgr[:, 1]
            blue_channel = skin_pixels_bgr[:, 0]
            
            # Natural skin has specific red-green correlation
            rg_correlation = float(np.corrcoef(red_channel, green_channel)[0, 1])
            
            return {
                **color_analysis,
                'uniformity_score': uniformity_score,
                'color_temperature': color_temperature,
                'rg_correlation': rg_correlation,
                'skin_pixel_count': len(skin_pixels_bgr),
                'skin_coverage': float(len(skin_pixels_bgr) / face_color.size * 3),
                'is_unnaturally_uniform': uniformity_score < 50,
                'has_unnatural_color_temp': abs(color_temperature - 0.8) > 0.4,
                'artificial_skin_probability': float(1.0 / (1.0 + np.exp(-(0.7 - abs(rg_correlation)) * 10)))
            }
            
        except Exception as e:
            logger.error(f"Advanced skin analysis error: {e}")
            return {'error': str(e)}
    
    def _skin_segmentation_hsv(self, hsv):
        """Skin segmentation using HSV color space"""
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        lower_skin2 = np.array([170, 20, 70])
        upper_skin2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        return cv2.bitwise_or(mask1, mask2)
    
    def _skin_segmentation_yuv(self, yuv):
        """Skin segmentation using YUV color space"""
        lower_skin = np.array([0, 133, 77])
        upper_skin = np.array([255, 173, 127])
        return cv2.inRange(yuv, lower_skin, upper_skin)
    
    def _micro_expression_analysis(self, face_gray):
        """Analyze micro-expressions for authenticity"""
        try:
            # Optical flow analysis for subtle movements
            # Since we have a single frame, we'll analyze texture gradients
            # that would be affected by micro-expressions
            
            # Divide face into regions
            h, w = face_gray.shape
            
            # Upper face (forehead, eyebrows)
            upper_region = face_gray[:h//3, :]
            
            # Middle face (eyes, nose)
            middle_region = face_gray[h//3:2*h//3, :]
            
            # Lower face (mouth, chin)
            lower_region = face_gray[2*h//3:, :]
            
            regions = [upper_region, middle_region, lower_region]
            region_names = ['upper', 'middle', 'lower']
            
            region_analysis = {}
            
            for region, name in zip(regions, region_names):
                if region.size == 0:
                    continue
                
                # Gradient analysis for each region
                grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                gradient_direction = np.arctan2(grad_y, grad_x)
                
                # Analyze gradient patterns
                region_analysis[f'{name}_gradient_mean'] = float(np.mean(gradient_magnitude))
                region_analysis[f'{name}_gradient_std'] = float(np.std(gradient_magnitude))
                region_analysis[f'{name}_direction_coherence'] = float(np.std(gradient_direction))
            
            # Overall micro-expression indicators
            gradient_variations = [region_analysis.get(f'{name}_gradient_std', 0) for name in region_names]
            expression_naturalness = float(np.var(gradient_variations))
            
            return {
                **region_analysis,
                'expression_naturalness': expression_naturalness,
                'has_frozen_expression': expression_naturalness < self.thresholds['micro_expression_threshold']
            }
            
        except Exception as e:
            logger.error(f"Micro-expression analysis error: {e}")
            return {'error': str(e)}
    
    def _lighting_consistency_analysis(self, face_color):
        """Analyze lighting consistency across the face"""
        try:
            # Convert to LAB color space for better lighting analysis
            lab = cv2.cvtColor(face_color, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Divide face into quadrants
            h, w = l_channel.shape
            quadrants = [
                l_channel[:h//2, :w//2],      # Top-left
                l_channel[:h//2, w//2:],      # Top-right
                l_channel[h//2:, :w//2],      # Bottom-left
                l_channel[h//2:, w//2:]       # Bottom-right
            ]
            
            # Analyze lighting in each quadrant
            quadrant_means = [float(np.mean(q)) for q in quadrants]
            quadrant_stds = [float(np.std(q)) for q in quadrants]
            
            # Lighting consistency metrics
            lighting_variance = float(np.var(quadrant_means))
            lighting_gradient = float(abs(quadrant_means[0] - quadrant_means[3]))  # Diagonal difference
            
            # Shadow analysis
            shadow_threshold = np.percentile(l_channel, 20)
            shadow_mask = l_channel < shadow_threshold
            shadow_regions = measure.label(shadow_mask)
            shadow_count = len(np.unique(shadow_regions)) - 1  # Subtract background
            
            # Highlight analysis
            highlight_threshold = np.percentile(l_channel, 80)
            highlight_mask = l_channel > highlight_threshold
            highlight_regions = measure.label(highlight_mask)
            highlight_count = len(np.unique(highlight_regions)) - 1
            
            return {
                'quadrant_means': quadrant_means,
                'quadrant_stds': quadrant_stds,
                'lighting_variance': lighting_variance,
                'lighting_gradient': lighting_gradient,
                'shadow_count': shadow_count,
                'highlight_count': highlight_count,
                'lighting_inconsistency': lighting_variance > self.thresholds['lighting_consistency'],
                'unnatural_lighting_score': float(lighting_variance / 100 + abs(lighting_gradient) / 50)
            }
            
        except Exception as e:
            logger.error(f"Lighting consistency analysis error: {e}")
            return {'error': str(e)}
    
    def _frequency_domain_analysis(self, face_gray):
        """Advanced frequency domain analysis"""
        try:
            # 2D FFT
            f_transform = np.fft.fft2(face_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Analyze frequency distribution in rings
            max_radius = min(center_x, center_y)
            ring_energies = []
            
            for r in range(1, max_radius, max_radius // 10):
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), r, 1, thickness=2)
                ring_energy = np.sum(magnitude_spectrum * mask)
                ring_energies.append(ring_energy)
            
            # High frequency energy (potential compression artifacts)
            high_freq_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(high_freq_mask, (center_x, center_y), max_radius // 3, 0, -1)
            high_freq_mask = 1 - high_freq_mask
            high_freq_energy = float(np.sum(magnitude_spectrum * high_freq_mask))
            
            # Low frequency energy (overall structure)
            low_freq_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(low_freq_mask, (center_x, center_y), max_radius // 6, 1, -1)
            low_freq_energy = float(np.sum(magnitude_spectrum * low_freq_mask))
            
            # Frequency ratio analysis
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-6)
            
            # Spectral rolloff
            total_energy = np.sum(magnitude_spectrum)
            cumulative_energy = 0
            rolloff_freq = 0
            
            for i, energy in enumerate(ring_energies):
                cumulative_energy += energy
                if cumulative_energy >= 0.85 * total_energy:
                    rolloff_freq = i
                    break
            
            return {
                'high_freq_energy': high_freq_energy,
                'low_freq_energy': low_freq_energy,
                'freq_ratio': float(freq_ratio),
                'spectral_rolloff': rolloff_freq,
                'ring_energies': [float(e) for e in ring_energies],
                'has_compression_artifacts': high_freq_energy < 1000,
                'frequency_anomaly_score': float(abs(freq_ratio - 0.3) / 0.3)
            }
            
        except Exception as e:
            logger.error(f"Frequency domain analysis error: {e}")
            return {'error': str(e)}
    
    def _biometric_consistency_analysis(self, face_gray):
        """Analyze biometric consistency"""
        try:
            # Feature point detection using SIFT
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(face_gray, None)
            
            if descriptors is None or len(descriptors) == 0:
                return {'error': 'No feature points detected'}
            
            # Analyze feature point distribution
            kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            
            # Feature density analysis
            h, w = face_gray.shape
            feature_density = float(len(keypoints) / (h * w))
            
            # Spatial distribution analysis
            x_coords = kp_coords[:, 0]
            y_coords = kp_coords[:, 1]
            
            x_variance = float(np.var(x_coords))
            y_variance = float(np.var(y_coords))
            spatial_distribution = float(np.sqrt(x_variance + y_variance))
            
            # Feature strength analysis
            responses = [kp.response for kp in keypoints]
            mean_response = float(np.mean(responses))
            response_variance = float(np.var(responses))
            
            # Scale analysis
            scales = [kp.size for kp in keypoints]
            scale_variance = float(np.var(scales))
            
            return {
                'feature_count': len(keypoints),
                'feature_density': feature_density,
                'spatial_distribution': spatial_distribution,
                'mean_response': mean_response,
                'response_variance': response_variance,
                'scale_variance': scale_variance,
                'biometric_quality_score': float(feature_density * mean_response * spatial_distribution / 1000),
                'has_sufficient_features': len(keypoints) > 50
            }
            
        except Exception as e:
            logger.error(f"Biometric consistency analysis error: {e}")
            return {'error': str(e)}
    
    def ultra_precision_ensemble_detection(self, image_path, facial_analysis, quality_analysis):
        """Ultra-precision ensemble detection with advanced scoring"""
        try:
            suspicion_factors = []
            confidence_factors = []
            detailed_reasons = []
            technical_indicators = {}
            
            # Initialize scoring system
            base_score = 0.5
            confidence_adjustments = []
            
            # 1. Facial Analysis Scoring
            if facial_analysis.get('face_count', 0) == 0:
                suspicion_factors.append(0.6)
                detailed_reasons.append("❌ No human faces detected in image")
                confidence_adjustments.append(-0.2)
            elif facial_analysis.get('face_count', 0) > 4:
                suspicion_factors.append(0.3)
                detailed_reasons.append("⚠️ Unusually high number of faces detected")
            
            # 2. Advanced Face Analysis
            for i, face_detail in enumerate(facial_analysis.get('face_details', [])):
                face_suspicion = 0.0
                face_confidence_adj = 0.0
                
                # Geometric Analysis
                geometric = face_detail.get('geometric_analysis', {})
                if geometric.get('geometric_anomaly', False):
                    face_suspicion += 0.15
                    detailed_reasons.append(f"🔍 Face {i+1}: Geometric proportions anomaly detected")
                
                # Multi-method Symmetry Analysis
                symmetry = face_detail.get('symmetry_analysis', {})
                combined_symmetry = symmetry.get('combined_symmetry', 0)
                
                if combined_symmetry > self.thresholds['symmetry_perfect']:
                    face_suspicion += 0.35
                    detailed_reasons.append(f"⚠️ Face {i+1}: Unnaturally perfect symmetry ({combined_symmetry:.3f})")
                    face_confidence_adj += 0.15
                elif combined_symmetry > self.thresholds['symmetry_suspicious']:
                    face_suspicion += 0.20
                    detailed_reasons.append(f"🔍 Face {i+1}: Suspicious symmetry level ({combined_symmetry:.3f})")
                    face_confidence_adj += 0.1
                
                if symmetry.get('symmetry_variance', 1) < 0.01:
                    face_suspicion += 0.15
                    detailed_reasons.append(f"🔍 Face {i+1}: Extremely low symmetry variance")
                
                # Ultra Texture Analysis
                texture = face_detail.get('texture_analysis', {})
                if texture.get('is_overly_smooth', False):
                    face_suspicion += 0.30
                    detailed_reasons.append(f"⚠️ Face {i+1}: Unnaturally smooth skin texture")
                    face_confidence_adj += 0.12
                
                if texture.get('has_unnatural_uniformity', False):
                    face_suspicion += 0.25
                    detailed_reasons.append(f"🔍 Face {i+1}: Unnatural skin uniformity detected")
                
                artificial_prob = texture.get('artificial_texture_probability', 0)
                if artificial_prob > 0.7:
                    face_suspicion += 0.20
                    detailed_reasons.append(f"🔍 Face {i+1}: High artificial texture probability ({artificial_prob:.2f})")
                
                # Comprehensive Eye Analysis
                eye_analysis = face_detail.get('eye_analysis', {})
                if eye_analysis.get('eye_count', 0) != 2:
                    face_suspicion += 0.25
                    detailed_reasons.append(f"⚠️ Face {i+1}: Abnormal eye detection ({eye_analysis.get('eye_count', 0)} eyes)")
                
                consistency = eye_analysis.get('consistency_analysis', {})
                if consistency.get('overall_consistency', 1) < 0.6:
                    face_suspicion += 0.20
                    detailed_reasons.append(f"🔍 Face {i+1}: Poor eye consistency")
                
                # Advanced Skin Analysis
                skin = face_detail.get('skin_analysis', {})
                if skin.get('is_unnaturally_uniform', False):
                    face_suspicion += 0.25
                    detailed_reasons.append(f"🔍 Face {i+1}: Unnaturally uniform skin detected")
                
                if skin.get('artificial_skin_probability', 0) > 0.7:
                    face_suspicion += 0.20
                    detailed_reasons.append(f"⚠️ Face {i+1}: High artificial skin probability")
                
                # Micro-expression Analysis
                micro_expr = face_detail.get('micro_expression_analysis', {})
                if micro_expr.get('has_frozen_expression', False):
                    face_suspicion += 0.15
                    detailed_reasons.append(f"🔍 Face {i+1}: Frozen/unnatural expression detected")
                
                # Lighting Consistency
                lighting = face_detail.get('lighting_analysis', {})
                if lighting.get('lighting_inconsistency', False):
                    face_suspicion += 0.20
                    detailed_reasons.append(f"⚠️ Face {i+1}: Inconsistent lighting patterns")
                
                unnatural_lighting = lighting.get('unnatural_lighting_score', 0)
                if unnatural_lighting > 0.5:
                    face_suspicion += 0.15
                    detailed_reasons.append(f"🔍 Face {i+1}: Unnatural lighting score: {unnatural_lighting:.2f}")
                
                # Frequency Domain Analysis
                freq_analysis = face_detail.get('frequency_analysis', {})
                if freq_analysis.get('has_compression_artifacts', False):
                    face_suspicion += 0.15
                    detailed_reasons.append(f"🔍 Face {i+1}: Compression artifacts in frequency domain")
                
                freq_anomaly = freq_analysis.get('frequency_anomaly_score', 0)
                if freq_anomaly > 0.5:
                    face_suspicion += 0.10
                    detailed_reasons.append(f"🔍 Face {i+1}: Frequency domain anomaly")
                
                # Biometric Analysis
                biometric = face_detail.get('biometric_analysis', {})
                if not biometric.get('has_sufficient_features', True):
                    face_suspicion += 0.15
                    detailed_reasons.append(f"⚠️ Face {i+1}: Insufficient biometric features")
                
                biometric_quality = biometric.get('biometric_quality_score', 0)
                if biometric_quality < 0.1:
                    face_suspicion += 0.10
                    detailed_reasons.append(f"🔍 Face {i+1}: Low biometric quality score")
                
                suspicion_factors.append(face_suspicion)
                confidence_adjustments.append(face_confidence_adj)
                
                # Store technical indicators
                technical_indicators[f'face_{i+1}'] = {
                    'symmetry_score': combined_symmetry,
                    'texture_complexity': texture.get('texture_complexity_score', 0),
                    'artificial_texture_prob': artificial_prob,
                    'skin_uniformity': skin.get('uniformity_score', 0),
                    'lighting_score': unnatural_lighting,
                    'biometric_quality': biometric_quality
                }
            
            # 3. Global Quality Analysis
            quality_score = quality_analysis.get('overall_quality_score', 0.5)
            if quality_score < 0.2:
                suspicion_factors.append(0.30)
                detailed_reasons.append("⚠️ Extremely poor image quality detected")
                confidence_adjustments.append(-0.15)
            elif quality_score < 0.4:
                suspicion_factors.append(0.15)
                detailed_reasons.append("🔍 Poor image quality may indicate manipulation")
            
            # Advanced blur analysis
            blur_scores = quality_analysis.get('blur_scores', {})
            avg_blur = np.mean(list(blur_scores.values())) if blur_scores else 0
            if avg_blur < 50:
                suspicion_factors.append(0.20)
                detailed_reasons.append("⚠️ Significant blur detected - possible processing artifact")
            
            # Compression analysis
            compression = quality_analysis.get('compression_analysis', {})
            if compression.get('has_significant_artifacts', False):
                suspicion_factors.append(0.25)
                detailed_reasons.append("🔍 Significant compression artifacts detected")
                
            blocking_score = compression.get('blocking_score', 0)
            if blocking_score > 0.4:
                suspicion_factors.append(0.15)
                detailed_reasons.append(f"🔍 JPEG blocking artifacts detected (score: {blocking_score:.2f})")
            
            # Color analysis
            color_analysis = quality_analysis.get('color_analysis', {})
            if color_analysis.get('color_inconsistency', False):
                suspicion_factors.append(0.20)
                detailed_reasons.append("⚠️ Color inconsistencies across image regions")
            
            # Edge analysis
            edge_analysis = quality_analysis.get('edge_analysis', {})
            if edge_analysis.get('edge_anomaly', False):
                suspicion_factors.append(0.15)
                detailed_reasons.append("🔍 Edge pattern anomalies detected")
            
            # 4. Calculate Final Scores with Advanced Weighting
            total_suspicion = sum(suspicion_factors)
            weighted_suspicion = min(1.0, total_suspicion * 0.85)  # Slight dampening for precision
            
            base_confidence = 0.75
            confidence_adjustment = sum(confidence_adjustments) * 0.6
            final_confidence = min(0.98, max(0.55, base_confidence + confidence_adjustment))
            
            # 5. Advanced Classification with Multiple Thresholds
            if weighted_suspicion >= 0.75:
                prediction = "FAKE"
                confidence = min(0.97, 0.70 + weighted_suspicion * 0.27)
                risk_level = "CRITICAL"
                threat_level = "🚨 HIGH THREAT"
            elif weighted_suspicion >= 0.55:
                prediction = "HIGHLY_SUSPICIOUS"
                confidence = 0.75 + (weighted_suspicion - 0.55) * 0.15
                risk_level = "HIGH"
                threat_level = "⚠️ HIGH RISK"
            elif weighted_suspicion >= 0.35:
                prediction = "SUSPICIOUS"
                confidence = 0.65 + (weighted_suspicion - 0.35) * 0.25
                risk_level = "MEDIUM"
                threat_level = "🔍 MEDIUM RISK"
            elif weighted_suspicion >= 0.15:
                prediction = "LIKELY_REAL"
                confidence = 0.70 + (0.35 - weighted_suspicion) * 0.15
                risk_level = "LOW"
                threat_level = "✅ LOW RISK"
            else:
                prediction = "AUTHENTIC"
                confidence = min(0.96, 0.80 + (0.15 - weighted_suspicion) * 0.8)
                risk_level = "MINIMAL"
                threat_level = "✅ AUTHENTIC"
            
            # 6. Generate Comprehensive Analysis Report
            analysis_summary = {
                'total_indicators': len(suspicion_factors),
                'high_risk_indicators': len([s for s in suspicion_factors if s > 0.2]),
                'medium_risk_indicators': len([s for s in suspicion_factors if 0.1 <= s <= 0.2]),
                'low_risk_indicators': len([s for s in suspicion_factors if s < 0.1]),
                'confidence_boosters': len([c for c in confidence_adjustments if c > 0]),
                'primary_concerns': [r for r in detailed_reasons if '⚠️' in r or '🚨' in r],
                'secondary_concerns': [r for r in detailed_reasons if '🔍' in r]
            }
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'suspicion_score': weighted_suspicion,
                'risk_level': risk_level,
                'threat_level': threat_level,
                'detailed_reasons': detailed_reasons,
                'analysis_summary': analysis_summary,
                'technical_indicators': technical_indicators,
                'suspicion_factors': suspicion_factors,
                'confidence_adjustments': confidence_adjustments,
                'analysis_depth': 'ultra_precision_ensemble',
                'model_version': 'UltraPrecision-v2.0',
                'processing_quality': 'maximum_precision'
            }
            
        except Exception as e:
            logger.error(f"Ultra precision ensemble detection error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'analysis_depth': 'ultra_precision_ensemble'
            }
    
    def analyze_image(self, image_path):
        """Main ultra-precision analysis function"""
        try:
            logger.info(f"Starting ultra-precision analysis for: {image_path}")
            
            # Ultra-comprehensive facial analysis
            facial_analysis = self.ultra_facial_analysis(image_path)
            
            # Advanced quality analysis (reuse from previous implementation)
            quality_analysis = self.advanced_quality_analysis(image_path)
            
            # Ultra-precision ensemble detection
            detection_result = self.ultra_precision_ensemble_detection(image_path, facial_analysis, quality_analysis)
            
            # Compile ultra-comprehensive result
            result = {
                'prediction': detection_result['prediction'],
                'confidence': detection_result['confidence'],
                'risk_level': detection_result['risk_level'],
                'threat_level': detection_result['threat_level'],
                'suspicion_score': detection_result['suspicion_score'],
                'has_faces': facial_analysis.get('face_count', 0) > 0,
                'face_count': facial_analysis.get('face_count', 0),
                'faces': facial_analysis.get('faces', []),
                'facial_analysis': facial_analysis,
                'quality_analysis': quality_analysis,
                'detection_reasons': detection_result.get('detailed_reasons', []),
                'analysis_summary': detection_result.get('analysis_summary', {}),
                'technical_indicators': detection_result.get('technical_indicators', {}),
                'model_used': 'ultra_precision_ensemble_cv',
                'model_version': detection_result.get('model_version', 'UltraPrecision-v2.0'),
                'analysis_depth': 'ultra_comprehensive',
                'processing_quality': 'maximum_precision'
            }
            
            # Generate advanced risk assessment
            risk_factors = []
            primary_concerns = detection_result.get('analysis_summary', {}).get('primary_concerns', [])
            secondary_concerns = detection_result.get('analysis_summary', {}).get('secondary_concerns', [])
            
            if not result['has_faces']:
                risk_factors.append("No human faces detected")
            if len(primary_concerns) > 0:
                risk_factors.append(f"{len(primary_concerns)} critical indicators found")
            if len(secondary_concerns) > 3:
                risk_factors.append(f"{len(secondary_concerns)} suspicious patterns detected")
            if detection_result['confidence'] < 0.7:
                risk_factors.append("Low confidence in prediction")
            if quality_analysis.get('overall_quality_score', 0.5) < 0.3:
                risk_factors.append("Poor image quality affects analysis")
            
            result['risk_factors'] = risk_factors
            result['analysis_confidence'] = 'HIGH' if detection_result['confidence'] > 0.8 else 'MEDIUM' if detection_result['confidence'] > 0.6 else 'LOW'
            
            logger.info(f"Ultra-precision analysis complete: {result['prediction']} (confidence: {result['confidence']:.3f}, threat: {result['threat_level']})")
            return result
            
        except Exception as e:
            logger.error(f"Ultra-precision analysis error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'model_used': 'ultra_precision_ensemble_cv',
                'analysis_depth': 'error'
            }
    
    def advanced_quality_analysis(self, image_path):
        """Enhanced image quality analysis (reusing from previous implementation)"""
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
            
            block_size = 8
            artifact_scores = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    if block.shape == (block_size, block_size):
                        dct_block = cv2.dct(block)
                        high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                        artifact_scores.append(high_freq)
            
            if artifact_scores:
                mean_artifact = np.mean(artifact_scores)
                std_artifact = np.std(artifact_scores)
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
            h, w = gray.shape
            block_boundaries_h = []
            block_boundaries_v = []
            
            for i in range(8, h-8, 8):
                diff = np.mean(np.abs(gray[i-1, :] - gray[i, :]))
                block_boundaries_h.append(diff)
            
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
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
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
            canny = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            edge_density = np.sum(canny > 0) / (canny.shape[0] * canny.shape[1])
            edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_mean = np.mean(edge_strength)
            edge_std = np.std(edge_strength)
            
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
            blur_score = np.mean(list(blur_scores.values()))
            normalized_blur = min(1.0, blur_score / 1000)
            normalized_noise = min(1.0, noise_level / 10000)
            compression_score = compression_analysis.get('mean_artifact_score', 0)
            normalized_compression = min(1.0, compression_score / 2000)
            
            quality_score = (0.4 * normalized_blur + 0.3 * (1 - normalized_noise) + 0.3 * (1 - normalized_compression))
            return float(max(0, min(1, quality_score)))
        except:
            return 0.5