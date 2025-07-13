import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button, Container, Typography, LinearProgress, Box, Paper, Chip, Alert } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SecurityIcon from '@mui/icons-material/Security';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { motion } from 'framer-motion';
import styled from 'styled-components';

// Styled components for Dark Design aesthetic
const BackgroundContainer = styled.div`
  background-image: url('/design-background.jpg');
  background-size: cover;
  background-position: center;
  min-height: 100vh;
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const Overlay = styled.div`
  background: rgba(13, 13, 13, 0.8);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 20px;
  text-align: center;
  color: white;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const NeonButton = styled(Button)`
  background: linear-gradient(135deg, #00ffcc, #00bfff);
  color: white;
  border: none;
  border-radius: 25px;
  padding: 10px 20px;
  font-size: 16px;
  font-weight: bold;
  text-transform: none;
  transition: transform 0.2s, box-shadow 0.2s;

  &:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(0, 255, 204, 0.4);
  }
`;

const AnimatedBox = motion(Box);

function Upload() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiData, setApiData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Test backend connection with better error handling
    const testBackend = async () => {
      try {
        const response = await axios.get("https://sheguard.onrender.com/health", {
          timeout: 10000 // 10 second timeout
        });
        setApiData(response.data);
        console.log("Backend connected successfully:", response.data);
      } catch (error) {
        console.error("Backend connection failed:", error);
        setError("Backend service is currently unavailable. Please try again later.");
      }
    };
    
    testBackend();
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setIsLoading(true);
    setMessage('');
    setAnalysisResult(null);
    setError(null);

    try {
      // Upload file
      const uploadResponse = await axios.post('https://sheguard.onrender.com/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000 // 30 second timeout for file upload
      });
      setMessage(uploadResponse.data.message);

      // Analyze file
      const analysisResponse = await axios.post('https://sheguard.onrender.com/analyze', {
        file_path: uploadResponse.data.file_path,
      }, {
        timeout: 60000 // 60 second timeout for analysis
      });
      setAnalysisResult(analysisResponse.data);
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        setError('Request timed out. The backend service might be starting up. Please try again in a few minutes.');
      } else if (error.response?.status === 502 || error.response?.status === 503) {
        setError('Backend service is starting up. Please wait a few minutes and try again.');
      } else {
        setError(`Error: ${error.message}. Please try again.`);
      }
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW': return '#4caf50';
      case 'MEDIUM': return '#ff9800';
      case 'HIGH': return '#f44336';
      default: return '#757575';
    }
  };

  const getPredictionIcon = (prediction) => {
    if (prediction === 'REAL') return <CheckCircleIcon style={{ color: '#4caf50' }} />;
    if (prediction === 'FAKE') return <WarningIcon style={{ color: '#f44336' }} />;
    return <SecurityIcon style={{ color: '#757575' }} />;
  };

  return (
    <BackgroundContainer>
      <Container maxWidth="sm">
        <Overlay>
          <Typography variant="h4" gutterBottom style={{ color: '#00ffcc', fontWeight: 'bold' }}>
            SheGuard
          </Typography>
          <Typography variant="subtitle1" gutterBottom style={{ color: '#b3b3b3' }}>
            Upload an image to detect if it's a deepfake.
          </Typography>

          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ display: 'none' }}
            id="file-upload"
          />
          <label htmlFor="file-upload">
            <NeonButton
              variant="contained"
              component="span"
              startIcon={<CloudUploadIcon />}
              style={{ marginBottom: '20px' }}
            >
              Choose File
            </NeonButton>
          </label>

          {file && (
            <Typography variant="body1" style={{ marginBottom: '20px', color: '#b3b3b3' }}>
              Selected file: {file.name}
            </Typography>
          )}

          <NeonButton
            variant="contained"
            onClick={handleUpload}
            disabled={isLoading || !file}
            fullWidth
          >
            {isLoading ? 'Analyzing...' : 'Upload and Analyze'}
          </NeonButton>

          {isLoading && (
            <AnimatedBox
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              style={{ marginTop: '20px' }}
            >
              <LinearProgress />
            </AnimatedBox>
          )}

          {message && (
            <Typography variant="body1" style={{ marginTop: '20px', color: '#b3b3b3' }}>
              {message}
            </Typography>
          )}

          {error && (
            <Alert severity="error" style={{ marginTop: '20px' }}>
              {error}
            </Alert>
          )}

          {analysisResult && (
            <AnimatedBox
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              style={{ marginTop: '20px' }}
            >
              <Paper elevation={2} style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.1)' }}>
                <Typography variant="h6" style={{ color: '#00ffcc' }}>
                  🔍 Analysis Results
                </Typography>
                
                <Box style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                  {getPredictionIcon(analysisResult.prediction)}
                  <Typography variant="h6" style={{ marginLeft: '10px', color: '#ffffff' }}>
                    {analysisResult.prediction}
                  </Typography>
                  <Chip 
                    label={`${(analysisResult.confidence * 100).toFixed(1)}% confidence`}
                    style={{ marginLeft: '10px', backgroundColor: '#00ffcc', color: '#000' }}
                    size="small"
                  />
                </Box>

                {analysisResult.prediction === 'SUSPICIOUS' && (
                  <Alert severity="warning" style={{ marginBottom: '15px' }}>
                    This image shows suspicious characteristics that may indicate manipulation.
                  </Alert>
                )}

                <Box style={{ marginBottom: '15px' }}>
                  <Typography variant="body2" style={{ color: '#b3b3b3', marginBottom: '5px' }}>
                    Risk Level:
                  </Typography>
                  <Chip 
                    label={analysisResult.risk_level}
                    style={{ 
                      backgroundColor: getRiskColor(analysisResult.risk_level),
                      color: '#ffffff',
                      fontWeight: 'bold'
                    }}
                  />
                </Box>

                {analysisResult.has_faces && (
                  <Typography variant="body2" style={{ color: '#b3b3b3', marginBottom: '5px' }}>
                    ✅ {analysisResult.face_count} face(s) detected
                  </Typography>
                )}

                {analysisResult.detection_reasons && analysisResult.detection_reasons.length > 0 && (
                  <Box style={{ marginTop: '10px' }}>
                    <Typography variant="body2" style={{ color: '#00ffcc', marginBottom: '5px' }}>
                      🔍 Detection Analysis:
                    </Typography>
                    {analysisResult.detection_reasons.map((reason, index) => (
                      <Typography key={index} variant="body2" style={{ color: '#b3b3b3', marginLeft: '10px' }}>
                        • {reason}
                      </Typography>
                    ))}
                  </Box>
                )}

                {analysisResult.risk_factors && analysisResult.risk_factors.length > 0 && (
                  <Box style={{ marginTop: '10px' }}>
                    <Typography variant="body2" style={{ color: '#ff9800', marginBottom: '5px' }}>
                      ⚠️ Risk Factors:
                    </Typography>
                    {analysisResult.risk_factors.map((factor, index) => (
                      <Typography key={index} variant="body2" style={{ color: '#b3b3b3', marginLeft: '10px' }}>
                        • {factor}
                      </Typography>
                    ))}
                  </Box>
                )}

                <Typography variant="caption" style={{ color: '#757575', marginTop: '10px', display: 'block' }}>
                  Detection Method: {analysisResult.model_used} | Suspicion Score: {(analysisResult.suspicion_score * 100).toFixed(1)}%
                </Typography>
              </Paper>
            </AnimatedBox>
          )}

          {apiData && (
            <Typography variant="body1" style={{ marginTop: '20px', color: '#b3b3b3' }}>
              API Data: {JSON.stringify(apiData, null, 2)}
            </Typography>
          )}
        </Overlay>
      </Container>
    </BackgroundContainer>
  );
}

export default Upload;