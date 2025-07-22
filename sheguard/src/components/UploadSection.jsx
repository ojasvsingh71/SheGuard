import React, { useState, useRef } from 'react';
import { Upload, Image, AlertCircle, CheckCircle, XCircle, Eye, Shield, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const UploadSection = ({ onAnalysisResult, isAnalyzing, setIsAnalyzing }) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const API_BASE_URL = 'http://127.0.0.1:5000';

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setError('Please select a valid image file (JPEG, PNG, GIF, BMP, WebP)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      // Upload file
      const formData = new FormData();
      formData.append('file', selectedFile);

      const uploadResponse = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      // Analyze the uploaded file
      const analyzeResponse = await axios.post(`${API_BASE_URL}/analyze`, {
        file_path: uploadResponse.data.file_path,
      }, {
        timeout: 30000,
      });

      setResult(analyzeResponse.data);
      onAnalysisResult(analyzeResponse.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(
        err.response?.data?.error || 
        err.message || 
        'Failed to analyze image. Please try again.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getPredictionIcon = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case 'real': return <CheckCircle className="h-6 w-6 text-green-500" />;
      case 'fake': return <XCircle className="h-6 w-6 text-red-500" />;
      case 'suspicious': return <AlertCircle className="h-6 w-6 text-yellow-500" />;
      default: return <AlertCircle className="h-6 w-6 text-gray-500" />;
    }
  };

  return (
    <section className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-12"
      >
        <h2 className="text-4xl font-bold text-white mb-4">
          Upload & Analyze Media
        </h2>
        <p className="text-xl text-white/80 max-w-2xl mx-auto">
          Upload an image to detect potential deepfakes and manipulation using our advanced AI detection system
        </p>
      </motion.div>

      <div className="glass-effect rounded-3xl p-8 shadow-2xl">
        {!selectedFile ? (
          <motion.div
            className={`upload-area border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
              dragOver 
                ? 'border-white/50 bg-white/10' 
                : 'border-white/30 hover:border-white/50 hover:bg-white/5'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <div className="flex flex-col items-center space-y-6">
              <div className="bg-white/10 rounded-full p-6">
                <Upload className="h-12 w-12 text-white" />
              </div>
              
              <div>
                <h3 className="text-2xl font-semibold text-white mb-2">
                  Drop your image here
                </h3>
                <p className="text-white/70 mb-6">
                  or click to browse files
                </p>
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-white text-indigo-600 px-8 py-3 rounded-full font-semibold hover:bg-white/90 transition-colors duration-200 flex items-center space-x-2 mx-auto"
                >
                  <Image className="h-5 w-5" />
                  <span>Choose File</span>
                </motion.button>
              </div>
              
              <p className="text-sm text-white/60">
                Supports JPEG, PNG, GIF, BMP, WebP • Max size: 10MB
              </p>
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </motion.div>
        ) : (
          <div className="space-y-6">
            {/* Image Preview */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative bg-white/10 rounded-2xl p-4"
            >
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full max-h-96 object-contain rounded-xl"
              />
              <button
                onClick={resetUpload}
                className="absolute top-6 right-6 bg-red-500/80 hover:bg-red-500 text-white rounded-full p-2 transition-colors duration-200"
              >
                <XCircle className="h-5 w-5" />
              </button>
            </motion.div>

            {/* File Info */}
            <div className="bg-white/10 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Image className="h-5 w-5 text-white/70" />
                  <span className="text-white font-medium">{selectedFile.name}</span>
                </div>
                <span className="text-white/70 text-sm">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
            </div>

            {/* Analyze Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={analyzeImage}
              disabled={isAnalyzing}
              className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {isAnalyzing ? (
                <>
                  <div className="loading-spinner"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Shield className="h-5 w-5" />
                  <span>Analyze for Deepfakes</span>
                </>
              )}
            </motion.button>
          </div>
        )}

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6 bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center space-x-3"
            >
              <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
              <p className="text-red-200">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Display */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              className="mt-8 result-card"
            >
              <div className="bg-white rounded-2xl p-8 shadow-xl">
                <div className="text-center mb-8">
                  <div className="flex items-center justify-center space-x-3 mb-4">
                    {getPredictionIcon(result.prediction)}
                    <h3 className="text-3xl font-bold text-gray-800">
                      {result.prediction === 'REAL' ? 'Authentic Image' : 
                       result.prediction === 'FAKE' ? 'Deepfake Detected' : 
                       'Suspicious Content'}
                    </h3>
                  </div>
                  
                  <div className="flex items-center justify-center space-x-4 mb-6">
                    <div className="text-center">
                      <p className="text-sm text-gray-600 mb-1">Confidence</p>
                      <p className="text-2xl font-bold text-indigo-600">
                        {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    
                    {result.risk_level && (
                      <div className="text-center">
                        <p className="text-sm text-gray-600 mb-1">Risk Level</p>
                        <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${getRiskColor(result.risk_level)}`}>
                          {result.risk_level}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Detection Details */}
                  <div className="bg-gray-50 rounded-xl p-6">
                    <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
                      <Eye className="h-5 w-5 mr-2" />
                      Detection Details
                    </h4>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Faces Detected:</span>
                        <span className="font-medium">{result.face_count || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Model Used:</span>
                        <span className="font-medium">Advanced CV</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Suspicion Score:</span>
                        <span className="font-medium">
                          {(result.suspicion_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Factors */}
                  <div className="bg-gray-50 rounded-xl p-6">
                    <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
                      <AlertCircle className="h-5 w-5 mr-2" />
                      Risk Factors
                    </h4>
                    <div className="space-y-2">
                      {result.risk_factors && result.risk_factors.length > 0 ? (
                        result.risk_factors.map((factor, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                            <span className="text-sm text-gray-700">{factor}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-gray-600">No significant risk factors detected</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Quality Metrics */}
                {result.quality_metrics && (
                  <div className="mt-6 bg-gray-50 rounded-xl p-6">
                    <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
                      <Zap className="h-5 w-5 mr-2" />
                      Image Quality Analysis
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <p className="text-sm text-gray-600">Blur Score</p>
                        <p className="font-semibold">{result.quality_metrics.blur_score?.toFixed(1) || 'N/A'}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-sm text-gray-600">Brightness</p>
                        <p className="font-semibold">{result.quality_metrics.mean_brightness?.toFixed(1) || 'N/A'}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-sm text-gray-600">Edge Density</p>
                        <p className="font-semibold">{(result.quality_metrics.edge_density * 100)?.toFixed(1) || 'N/A'}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-sm text-gray-600">Contrast</p>
                        <p className="font-semibold">{result.quality_metrics.brightness_std?.toFixed(1) || 'N/A'}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="mt-8 flex flex-col sm:flex-row gap-4">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={resetUpload}
                    className="flex-1 bg-indigo-600 text-white py-3 rounded-xl font-semibold hover:bg-indigo-700 transition-colors duration-200"
                  >
                    Analyze Another Image
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="flex-1 bg-gray-200 text-gray-800 py-3 rounded-xl font-semibold hover:bg-gray-300 transition-colors duration-200"
                  >
                    Download Report
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
};

export default UploadSection;