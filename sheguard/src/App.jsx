import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import Hero from './components/Hero';
import UploadSection from './components/UploadSection';
import Features from './components/Features';
import Stats from './components/Stats';
import Footer from './components/Footer';

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500">
      <div className="relative">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%23ffffff%22%20fill-opacity%3D%220.05%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%222%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-20"></div>
        <Header />
        
        <main className="relative z-10">
          <Hero />
          
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="container mx-auto px-4 py-16"
          >
            <UploadSection 
              onAnalysisResult={setAnalysisResult}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          </motion.div>

          <Features />
          <Stats />
        </main>
        
        <Footer />
      </div>
    </div>
  )
  ;
}

export default App;