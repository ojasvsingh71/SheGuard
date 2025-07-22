import React from 'react';
import { motion } from 'framer-motion';

const Stats = () => {
  const stats = [
    {
      number: '99.2%',
      label: 'Detection Accuracy',
      description: 'Industry-leading precision in identifying deepfakes'
    },
    {
      number: '2.3M+',
      label: 'Images Analyzed',
      description: 'Trusted by users worldwide for media verification'
    },
    {
      number: '<3s',
      label: 'Analysis Time',
      description: 'Lightning-fast results for real-time detection'
    },
    {
      number: '24/7',
      label: 'Availability',
      description: 'Always online and ready to protect you'
    }
  ];

  return (
    <section className="py-20 relative">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
            Trusted by Millions
            <span className="block bg-gradient-to-r from-green-300 to-blue-300 bg-clip-text text-transparent">
              Worldwide
            </span>
          </h2>
          <p className="text-xl text-white/80 max-w-3xl mx-auto">
            Join the global community fighting against digital deception and misinformation
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.5 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ 
                duration: 0.6, 
                delay: index * 0.1,
                type: "spring",
                stiffness: 100
              }}
              viewport={{ once: true }}
              className="glass-effect rounded-2xl p-8 text-center group hover:scale-105 transition-transform duration-300"
            >
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                transition={{ duration: 0.8, delay: index * 0.1 + 0.3 }}
                viewport={{ once: true }}
                className="text-5xl lg:text-6xl font-bold bg-gradient-to-r from-yellow-300 via-pink-300 to-purple-300 bg-clip-text text-transparent mb-4 stats-counter"
              >
                {stat.number}
              </motion.div>
              
              <h3 className="text-xl font-bold text-white mb-2 group-hover:text-yellow-300 transition-colors duration-300">
                {stat.label}
              </h3>
              
              <p className="text-white/70 text-sm leading-relaxed">
                {stat.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Trust Indicators */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <div className="glass-effect rounded-2xl p-8 max-w-4xl mx-auto">
            <h3 className="text-2xl font-bold text-white mb-6">
              Recognized by Industry Leaders
            </h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 items-center opacity-60">
              {/* Placeholder for partner logos */}
              <div className="bg-white/10 rounded-lg p-4 h-16 flex items-center justify-center">
                <span className="text-white font-semibold">TechCrunch</span>
              </div>
              <div className="bg-white/10 rounded-lg p-4 h-16 flex items-center justify-center">
                <span className="text-white font-semibold">MIT Review</span>
              </div>
              <div className="bg-white/10 rounded-lg p-4 h-16 flex items-center justify-center">
                <span className="text-white font-semibold">IEEE</span>
              </div>
              <div className="bg-white/10 rounded-lg p-4 h-16 flex items-center justify-center">
                <span className="text-white font-semibold">AI Ethics</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Stats;