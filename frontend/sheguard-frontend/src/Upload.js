import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button, Container, Typography, LinearProgress, Box, Paper } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
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
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiData, setApiData] = useState(null);

  useEffect(() => {
    axios.get("https://sheguard.onrender.com/api")
      .then(response => {
        setApiData(response.data);
      })
      .catch(error => {
        console.error("There was an error fetching the data!", error);
      });
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
    setPrediction(null);

    try {
      // Upload file
      const uploadResponse = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(uploadResponse.data.message);

      // Analyze file
      const analysisResponse = await axios.post('http://127.0.0.1:5000/analyze', {
        file_path: uploadResponse.data.file_path,
      });
      setPrediction(analysisResponse.data.prediction);
    } catch (error) {
      setMessage('Error uploading file.');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
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

          {prediction && (
            <AnimatedBox
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              style={{ marginTop: '20px' }}
            >
              <Paper elevation={2} style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.1)' }}>
                <Typography variant="h6" style={{ color: '#00ffcc' }}>
                  Analysis Results
                </Typography>
                <Typography variant="body1" style={{ color: '#b3b3b3' }}>
                  Prediction: {prediction}
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