import React from 'react';
import '../styles/About.css';

function About() {
  return (
    <div className="about-container">
      <h1>About TradingView Integration Platform</h1>
      <div className="about-content">
        <section className="about-section">
          <h2>Our Mission</h2>
          <p>
            We aim to provide comprehensive market analysis and trading tools through
            an integrated platform that combines real-time market data with advanced
            technical analysis capabilities.
          </p>
        </section>

        <section className="about-section">
          <h2>Features</h2>
          <ul>
            <li>Real-time market data visualization</li>
            <li>Technical analysis tools and indicators</li>
            <li>Market overview and stock heatmaps</li>
            <li>Economic calendar integration</li>
            <li>Cryptocurrency market tracking</li>
            <li>Forex cross rates monitoring</li>
          </ul>
        </section>

        <section className="about-section">
          <h2>Technology Stack</h2>
          <p>
            Built using modern technologies including React, Python, and TradingView's
            advanced widgets, our platform delivers a seamless and powerful trading
            analysis experience.
          </p>
        </section>
      </div>
    </div>
  );
}

export default About;
