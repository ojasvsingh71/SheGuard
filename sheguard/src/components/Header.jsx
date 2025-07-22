import React from 'react';
import { Shield, Menu, X } from 'lucide-react';
import { motion } from 'framer-motion';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  return (
    <motion.header 
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="relative z-50"
    >
      <nav className="glass-effect border-b border-white/20">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <motion.div 
              className="flex items-center space-x-3"
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <div className="relative">
                <Shield className="h-8 w-8 text-white" />
                <div className="absolute inset-0 bg-white/20 rounded-full blur-xl"></div>
              </div>
              <span className="text-2xl font-bold text-white">SheGuard</span>
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-white/90 hover:text-white transition-colors duration-200 font-medium">
                Features
              </a>
              <a href="#how-it-works" className="text-white/90 hover:text-white transition-colors duration-200 font-medium">
                How It Works
              </a>
              <a href="#about" className="text-white/90 hover:text-white transition-colors duration-200 font-medium">
                About
              </a>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-white/20 backdrop-blur-sm text-white px-6 py-2 rounded-full border border-white/30 hover:bg-white/30 transition-all duration-200 font-medium"
              >
                Get Started
              </motion.button>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden text-white p-2"
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>

          {/* Mobile Navigation */}
          {isMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden mt-4 pb-4 border-t border-white/20"
            >
              <div className="flex flex-col space-y-4 pt-4">
                <a href="#features" className="text-white/90 hover:text-white transition-colors duration-200 font-medium">
                  Features
                </a>
                <a href="#how-it-works" className="text-white/90 hover:text-white transition-colors duration-200 font-medium">
                  How It Works
                </a>
                <a href="#about" className="text-white/90 hover:text-white transition-colors duration-200 font-medium">
                  About
                </a>
                <button className="bg-white/20 backdrop-blur-sm text-white px-6 py-2 rounded-full border border-white/30 hover:bg-white/30 transition-all duration-200 font-medium text-left">
                  Get Started
                </button>
              </div>
            </motion.div>
          )}
        </div>
      </nav>
    </motion.header>
  );
};

export default Header;