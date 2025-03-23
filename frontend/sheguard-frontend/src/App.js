import React from 'react';
import Navbar from './Navbar';
import Upload from './Upload';
import Footer from './Footer';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import './App.css';

const darkDesignTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ffcc',
    },
    secondary: {
      main: '#ff00ff',
    },
    background: {
      default: '#0d0d0d',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b3b3b3',
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h4: {
      fontWeight: 'bold',
      color: '#00ffcc',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkDesignTheme}>
      <CssBaseline />
      <Navbar />
      <Upload />
      <Footer />
    </ThemeProvider>
  );
}

export default App;