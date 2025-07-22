import React from 'react';
import { Shield, Github, Twitter, Linkedin, Mail } from 'lucide-react';
import { motion } from 'framer-motion';

const Footer = () => {
  return (
    <footer className="relative py-16 border-t border-white/10">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <motion.div 
              className="flex items-center space-x-3 mb-6"
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <div className="relative">
                <Shield className="h-8 w-8 text-white" />
                <div className="absolute inset-0 bg-white/20 rounded-full blur-xl"></div>
              </div>
              <span className="text-2xl font-bold text-white">SheGuard</span>
            </motion.div>
            
            <p className="text-white/80 mb-6 max-w-md leading-relaxed">
              Protecting digital integrity through advanced AI-powered deepfake detection. 
              Join us in the fight against misinformation and digital manipulation.
            </p>
            
            <div className="flex space-x-4">
              {[
                { icon: Github, href: 'https://github.com/ojasvsingh71/SheGuard' },
                { icon: Twitter, href: '#' },
                { icon: Linkedin, href: '#' },
                { icon: Mail, href: 'mailto:ojasvsingh0@gmail.com' }
              ].map((social, index) => (
                <motion.a
                  key={index}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.2, y: -2 }}
                  whileTap={{ scale: 0.9 }}
                  className="bg-white/10 hover:bg-white/20 p-3 rounded-full transition-colors duration-200"
                >
                  <social.icon className="h-5 w-5 text-white" />
                </motion.a>
              ))}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold text-lg mb-4">Quick Links</h3>
            <ul className="space-y-3">
              {['Features', 'How It Works', 'About Us', 'Contact'].map((link, index) => (
                <li key={index}>
                  <motion.a
                    href={`#${link.toLowerCase().replace(' ', '-')}`}
                    whileHover={{ x: 5 }}
                    className="text-white/70 hover:text-white transition-colors duration-200"
                  >
                    {link}
                  </motion.a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-white font-semibold text-lg mb-4">Resources</h3>
            <ul className="space-y-3">
              {['Documentation', 'API Reference', 'Privacy Policy', 'Terms of Service'].map((link, index) => (
                <li key={index}>
                  <motion.a
                    href="#"
                    whileHover={{ x: 5 }}
                    className="text-white/70 hover:text-white transition-colors duration-200"
                  >
                    {link}
                  </motion.a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-white/10 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-white/60 text-sm">
              © 2024 SheGuard. All rights reserved. Built with ❤️ for digital safety.
            </p>
            
            <div className="flex items-center space-x-6 text-sm text-white/60">
              <span>Made by Ojas Singh</span>
              <span>•</span>
              <span>Powered by AI</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;