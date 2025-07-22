import React from 'react';
import { Shield, Zap, Eye, Brain, Lock, Globe } from 'lucide-react';
import { motion } from 'framer-motion';

const Features = () => {
  const features = [
    {
      icon: Brain,
      title: 'Advanced AI Detection',
      description: 'State-of-the-art machine learning models trained on millions of images to identify even the most sophisticated deepfakes.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Get instant results with our optimized detection pipeline that processes images in seconds, not minutes.',
      color: 'from-yellow-500 to-orange-500'
    },
    {
      icon: Eye,
      title: 'Multi-layer Verification',
      description: 'Comprehensive analysis including face detection, quality assessment, and compression artifact detection.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Shield,
      title: 'High Accuracy',
      description: '99.2% accuracy rate with continuous model improvements and updates to stay ahead of emerging threats.',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: Lock,
      title: 'Privacy First',
      description: 'Your images are processed securely and never stored permanently. Complete privacy and data protection.',
      color: 'from-red-500 to-pink-500'
    },
    {
      icon: Globe,
      title: 'Global Impact',
      description: 'Join the fight against misinformation worldwide. Help create a safer digital environment for everyone.',
      color: 'from-indigo-500 to-purple-500'
    }
  ];

  return (
    <section id="features" className="py-20 relative">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
            Powerful Features for
            <span className="block bg-gradient-to-r from-yellow-300 to-pink-300 bg-clip-text text-transparent">
              Digital Safety
            </span>
          </h2>
          <p className="text-xl text-white/80 max-w-3xl mx-auto">
            Our comprehensive suite of AI-powered tools provides unmatched protection against deepfakes and manipulated media
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ y: -10 }}
              className="feature-card glass-effect rounded-2xl p-8 group"
            >
              <div className={`bg-gradient-to-r ${feature.color} rounded-2xl w-16 h-16 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="h-8 w-8 text-white" />
              </div>
              
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-yellow-300 transition-colors duration-300">
                {feature.title}
              </h3>
              
              <p className="text-white/80 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* How It Works Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mt-20"
          id="how-it-works"
        >
          <div className="text-center mb-12">
            <h3 className="text-3xl lg:text-4xl font-bold text-white mb-4">
              How SheGuard Works
            </h3>
            <p className="text-lg text-white/80 max-w-2xl mx-auto">
              Our three-step process ensures comprehensive analysis and accurate detection
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Upload & Process',
                description: 'Upload your image and our system immediately begins preprocessing and quality analysis.',
                icon: '📤'
              },
              {
                step: '02',
                title: 'AI Analysis',
                description: 'Advanced machine learning models analyze facial features, compression artifacts, and manipulation patterns.',
                icon: '🧠'
              },
              {
                step: '03',
                title: 'Results & Report',
                description: 'Receive detailed results with confidence scores, risk assessment, and comprehensive analysis report.',
                icon: '📊'
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="bg-white/10 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6 text-3xl">
                  {item.icon}
                </div>
                <div className="bg-gradient-to-r from-yellow-300 to-pink-300 bg-clip-text text-transparent text-sm font-bold mb-2">
                  STEP {item.step}
                </div>
                <h4 className="text-xl font-bold text-white mb-4">{item.title}</h4>
                <p className="text-white/70">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Features;