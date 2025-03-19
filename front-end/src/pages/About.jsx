import React from 'react';
import '../styles/About.css';

function About() {
  return (
    <div className="about-page black-theme">
      <div className="about-hero">
        <div className="container">
          <h1>About SPORS</h1>
          <p className="subtitle">Stock Prediction and Options Recommendation System</p>
        </div>
      </div>
      
      <div className="container">
        <div className="row">
          <div className="col-main">
            <div className="section-block">
              <h2>Our Mission</h2>
              <p>
                SPORS is designed to empower traders and investors with AI-driven insights for the 
                Indian stock market. By combining advanced Temporal Fusion Transformer (TFT) models 
                with intuitive analytics, we provide accurate price predictions and actionable 
                options strategy recommendations.
              </p>
              
              <h3>What is SPORS?</h3>
              <p>
                SPORS (Stock Prediction and Options Recommendation System) is an advanced financial 
                analytics platform that uses machine learning to predict stock price movements and 
                recommend appropriate options trading strategies. The platform bridges the gap between 
                complex technical analysis and practical trading decisions for both novice and experienced traders.
              </p>
            </div>

            <div className="section-block">
              <h3>Key Features</h3>
              <div className="features-container">
                <div className="feature-box">
                  <h4>Accurate Price Prediction</h4>
                  <p>
                    Our system employs state-of-the-art Temporal Fusion Transformer models that analyze 
                    historical price patterns, technical indicators, and market trends to generate 
                    precise price forecasts.
                  </p>
                </div>
                
                <div className="feature-box">
                  <h4>Options Strategy Recommendations</h4>
                  <p>
                    Based on predicted price movements, SPORS suggests appropriate options strategies, 
                    including specific strike prices for Call (CE) and Put (PE) options tailored to 
                    the forecasted market direction.
                  </p>
                </div>
                
                <div className="feature-box">
                  <h4>Custom Model Training</h4>
                  <p>
                    Users can train prediction models specific to individual stocks with customizable 
                    historical data ranges, allowing for personalized analysis based on different 
                    market conditions and timeframes.
                  </p>
                </div>
                
                <div className="feature-box">
                  <h4>Comprehensive Market Data</h4>
                  <p>
                    Integration with TradingView provides access to real-time market information, 
                    technical charts, economic calendars, and news to complement our predictive analytics.
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="col-sidebar">
            <div className="section-block">
              <h3>Implementation Details</h3>
              <p>
                Our platform combines cutting-edge technology with user-friendly design to deliver 
                powerful predictive analytics and actionable trading insights.
              </p>
            </div>
            
            <div className="section-block">
              <h3>How It Works</h3>
              <ol className="workflow-list">
                <li><span>Select a stock</span> from the NIFTY list</li>
                <li><span>View historical data</span> and analysis</li>
                <li><span>Train a custom model</span> or use pre-trained models</li>
                <li><span>Generate predictions</span> for future price movements</li>
                <li><span>Receive options strategies</span> with recommended strike prices</li>
                <li><span>Make informed trading decisions</span> based on AI insights</li>
              </ol>
            </div>
          </div>
        </div>
        
        <div className="row">
          <div className="col-full">
            <div className="section-block tech-framework-section">
              <h3>Technical Framework</h3>
              <p>
                SPORS is built on a robust, modern technology stack that enables accurate predictions 
                and responsive user experience across devices.
              </p>
              
              <div className="tech-grid">
                <div className="tech-category">
                  <h4>Front End</h4>
                  <ul className="tech-list">
                    <li>
                      <span className="tech-name">React.js</span>
                      <span className="tech-desc">Powers our interactive UI components and responsive design</span>
                    </li>
                    <li>
                      <span className="tech-name">TradingView Widgets</span>
                      <span className="tech-desc">Provides professional-grade market visualization tools</span>
                    </li>
                    <li>
                      <span className="tech-name">Custom CSS</span>
                      <span className="tech-desc">Ensures consistent styling and responsive layouts</span>
                    </li>
                  </ul>
                </div>
                
                <div className="tech-category">
                  <h4>Back End</h4>
                  <ul className="tech-list">
                    <li>
                      <span className="tech-name">FastAPI (Python)</span>
                      <span className="tech-desc">Delivers high-performance API endpoints with minimal latency</span>
                    </li>
                    <li>
                      <span className="tech-name">PyTorch</span>
                      <span className="tech-desc">Powers our deep learning prediction models</span>
                    </li>
                    <li>
                      <span className="tech-name">Temporal Fusion Transformers</span>
                      <span className="tech-desc">Advanced architecture for time-series forecasting with attention mechanism</span>
                    </li>
                  </ul>
                </div>
                
                <div className="tech-category">
                  <h4>Data Sources</h4>
                  <ul className="tech-list">
                    <li>
                      <span className="tech-name">Yahoo Finance</span>
                      <span className="tech-desc">Provides historical price data for model training and validation</span>
                    </li>
                    <li>
                      <span className="tech-name">BSE/NSE</span>
                      <span className="tech-desc">Supplies Indian market information and stock listings</span>
                    </li>
                    <li>
                      <span className="tech-name">TradingView API</span>
                      <span className="tech-desc">Enables live chart integration and technical analysis</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="row">
          <div className="col-full">
            <div className="section-block highlight-section">
              <h3>How TFT Models Work</h3>
              <p>
                The core of SPORS is its Temporal Fusion Transformer (TFT) architecture, a cutting-edge 
                deep learning model specifically designed for time-series forecasting tasks like stock 
                price prediction.
              </p>
              
              <div className="tft-explainer">
                <div className="tft-section">
                  <h5>1. Data Processing</h5>
                  <p>
                    Historical stock data including open, high, low, close prices and volume are 
                    collected, normalized, and transformed into sequential patterns.
                  </p>
                </div>
                
                <div className="tft-section">
                  <h5>2. Pattern Recognition</h5>
                  <p>
                    The TFT model identifies both short-term price movements and long-term market 
                    trends through its attention-based architecture.
                  </p>
                </div>
                
                <div className="tft-section">
                  <h5>3. Prediction Generation</h5>
                  <p>
                    Based on learned patterns, the model generates price predictions with built-in 
                    constraints to ensure realistic forecasts.
                  </p>
                </div>
                
                <div className="tft-section">
                  <h5>4. Options Strategy Formulation</h5>
                  <p>
                    Predicted price movements are analyzed to determine appropriate options strategies 
                    and optimal strike prices for maximum profit potential.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="row">
          <div className="col-full">
            <div className="section-block disclaimer-section">
              <h3>Disclaimer</h3>
              <p>
                SPORS is provided for informational and educational purposes only. The predictions and 
                recommendations generated by our system should not be considered as financial advice. 
                All investment decisions carry risk, and past performance is not indicative of future results.
              </p>
              <p>
                Options trading involves significant risk and is not suitable for all investors. Always 
                conduct your own research and consider consulting with a financial advisor before making 
                investment decisions based on SPORS recommendations.
              </p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .about-page.black-theme {
          color: #d1d4dc;
          background-color: #000000;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
          padding-bottom: 40px;
        }
        
        .about-hero {
          background-color: #121212;
          padding: 60px 0;
          margin-bottom: 40px;
          border-bottom: 1px solid #333;
          width: 100%;
        }
        
        .about-hero h1 {
          font-size: 3rem;
          color: white;
          margin-bottom: 15px;
          font-weight: 600;
        }
        
        .about-hero .subtitle {
          font-size: 1.4rem;
          color: #9598a1;
        }
        
        .container {
          width: 100%;
          max-width: 1400px;
          margin: 0 auto;
          padding: 0 20px;
        }
        
        .row {
          display: flex;
          flex-wrap: wrap;
          margin: 0 -15px 30px;
        }
        
        .col-main {
          flex: 0 0 66.666667%;
          max-width: 66.666667%;
          padding: 0 15px;
        }
        
        .col-sidebar {
          flex: 0 0 33.333333%;
          max-width: 33.333333%;
          padding: 0 15px;
        }
        
        .col-full {
          flex: 0 0 100%;
          max-width: 100%;
          padding: 0 15px;
        }
        
        .section-block {
          margin-bottom: 30px;
          padding-bottom: 15px;
        }
        
        .section-block h2 {
          color: white;
          font-size: 2rem;
          margin-bottom: 25px;
          border-bottom: 1px solid #333;
          padding-bottom: 15px;
        }
        
        .section-block h3 {
          color: white;
          font-size: 1.6rem;
          margin: 30px 0 20px;
        }
        
        .section-block p {
          line-height: 1.7;
          font-size: 1.05rem;
          color: #b0b3bc;
          margin-bottom: 20px;
        }
        
        .section-block h4 {
          color: #2962ff;
          font-size: 1.2rem;
          margin-bottom: 12px;
        }
        
        /* Tech Framework Section */
        .tech-framework-section {
          background-color: #121212;
          padding: 30px;
          border-radius: 6px;
          position: relative;
          margin-bottom: 40px;
        }
        
        .tech-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 30px;
          margin-top: 25px;
        }
        
        .tech-category {
          background-color: rgba(41, 98, 255, 0.03);
          border: 1px solid rgba(41, 98, 255, 0.1);
          border-radius: 6px;
          padding: 20px;
        }
        
        .tech-list {
          list-style: none;
          padding: 0;
          margin: 0;
        }
        
        .tech-list li {
          margin-bottom: 15px;
          display: flex;
          flex-direction: column;
        }
        
        .tech-name {
          color: #26a69a;
          font-weight: 600;
          font-size: 1.1rem;
          margin-bottom: 5px;
          display: block;
        }
        
        .tech-desc {
          color: #b0b3bc;
          font-size: 0.95rem;
        }
        
        /* Features */
        .features-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
          gap: 25px;
          margin-top: 25px;
        }
        
        .feature-box {
          background-color: #121212;
          border-left: 3px solid #2962ff;
          padding: 20px;
          border-radius: 4px;
        }
        
        .feature-box p {
          margin-bottom: 0;
        }
        
        /* Workflow */
        .workflow-list {
          counter-reset: workflow-counter;
          list-style-type: none;
          padding-left: 5px;
        }
        
        .workflow-list li {
          counter-increment: workflow-counter;
          position: relative;
          padding-left: 35px;
          margin-bottom: 15px;
          font-size: 1.05rem;
          color: #b0b3bc;
        }
        
        .workflow-list li:before {
          content: counter(workflow-counter);
          background-color: #2962ff;
          color: white;
          width: 25px;
          height: 25px;
          border-radius: 50%;
          display: inline-flex;
          justify-content: center;
          align-items: center;
          position: absolute;
          left: 0;
          top: 0;
          font-size: 14px;
          font-weight: bold;
        }
        
        .workflow-list li span {
          font-weight: 500;
          color: #d1d4dc;
        }
        
        /* TFT Explainer */
        .highlight-section {
          background-color: #121212;
          padding: 25px;
          border-radius: 6px;
          position: relative;
          overflow: hidden;
        }
        
        .highlight-section:before {
          content: "";
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 4px;
          background: linear-gradient(90deg, #2962ff, #3d5afe);
        }
        
        .tft-explainer {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 20px;
          margin-top: 30px;
        }
        
        .tft-section {
          background-color: #1a1a1a;
          padding: 20px;
          border-radius: 6px;
          border-top: 3px solid #2962ff;
        }
        
        .tft-section h5 {
          color: white;
          margin-bottom: 12px;
          font-size: 1.1rem;
        }
        
        .tft-section p {
          margin-bottom: 0;
        }
        
        /* Disclaimer */
        .disclaimer-section {
          background-color: #121212;
          padding: 25px;
          border-radius: 6px;
          border-left: 4px solid rgba(239, 83, 80, 0.7);
        }
        
        .disclaimer-section h3 {
          color: rgba(239, 83, 80, 0.9);
        }
        
        @media (max-width: 992px) {
          .tech-grid {
            grid-template-columns: 1fr;
          }
          
          .col-main, .col-sidebar {
            flex: 0 0 100%;
            max-width: 100%;
          }
          
          .features-container {
            grid-template-columns: 1fr;
          }
        }
        
        @media (max-width: 768px) {
          .about-hero {
            padding: 40px 0;
          }
          
          .about-hero h1 {
            font-size: 2.2rem;
          }
          
          .about-hero .subtitle {
            font-size: 1.2rem;
          }
          
          .section-block h2 {
            font-size: 1.8rem;
          }
          
          .tft-explainer {
            grid-template-columns: 1fr;
          }
          
          .tech-grid {
            grid-template-columns: 1fr;
            gap: 15px;
          }
        }
      `}</style>
    </div>
  );
}

export default About;
