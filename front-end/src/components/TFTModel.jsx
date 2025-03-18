import React, { useState } from 'react';
import { niftyStocks } from '../data/niftyStocks';

function TFTModel({ symbol }) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [activeAction, setActiveAction] = useState(null); // 'train' or 'predict'

  const API_BASE_URL = 'http://localhost:8000'; // Ensure this is consistently set to port 8000

  // Get the actual symbol from niftyStocks - optimized to always find the correct symbol
  const getActualSymbol = (displaySymbol) => {
    // First check if this is a pure symbol match
    const stockInfo = niftyStocks.find(stock => stock.symbol === displaySymbol);
    if (stockInfo) {
      console.log(`Symbol ${displaySymbol} matched to ${stockInfo.symbol} (${stockInfo.bseSymbol})`);
      return stockInfo.symbol; // Return display symbol for backend to map
    }
    
    // Check if it's a BSE:Symbol format
    if (displaySymbol.startsWith('BSE:')) {
      const cleanSymbol = displaySymbol.substring(4);
      const stockInfo = niftyStocks.find(stock => stock.symbol === cleanSymbol);
      if (stockInfo) {
        console.log(`BSE:Symbol ${displaySymbol} matched to ${stockInfo.symbol}`);
        return stockInfo.symbol;
      }
    }
    
    // If we can't find it, just return the original symbol
    console.log(`No match found for ${displaySymbol}, using as-is`);
    return displaySymbol;
  };

  const handleTrain = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);
      setActiveAction('train');

      const actualSymbol = getActualSymbol(symbol);
      console.log('Starting training for symbol:', actualSymbol);
      
      // Send only the display symbol - backend will map to BSE symbol
      const trainResponse = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: actualSymbol })
      });
      console.log('Training response status:', trainResponse.status);

      if (!trainResponse.ok) {
        const errorData = await trainResponse.json();
        console.error('Training failed:', errorData);
        throw new Error(errorData.detail || 'Training failed');
      }

      // Poll training progress
      console.log('Starting progress polling...');
      const progressInterval = setInterval(async () => {
        try {
          const progressResponse = await fetch(`${API_BASE_URL}/progress`);
          const data = await progressResponse.json();
          console.log('Progress update:', data.progress);
          setProgress(data.progress);
          
          if (data.progress === 100) {
            console.log('Training completed');
            clearInterval(progressInterval);
            setLoading(false);
          }
        } catch (err) {
          console.error('Error during progress check:', err);
          clearInterval(progressInterval);
          setLoading(false);
          setError(err.message);
        }
      }, 1000);

    } catch (err) {
      console.error('Error in handleTrain:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);
      setActiveAction('predict');

      const actualSymbol = getActualSymbol(symbol);
      console.log('Starting prediction for symbol:', actualSymbol);
      
      // Send only the display symbol - backend will map to BSE symbol
      const predictResponse = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: actualSymbol })
      });
      console.log('Prediction response status:', predictResponse.status);

      if (!predictResponse.ok) {
        const errorData = await predictResponse.json();
        console.error('Prediction failed:', errorData);
        throw new Error(errorData.detail || 'Prediction failed');
      }
      
      const result = await predictResponse.json();
      console.log('Received prediction:', result);
      setPrediction(result.prediction);
      setLoading(false);
    } catch (err) {
      console.error('Error in handlePredict:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="tradingview-widget-container">
      <div className="tft-widget">
        <h2>TFT Model Predictions</h2>
        <div className="tft-content">
          <div className="tft-actions-vertical">
            <div className="tft-action-block">
              <button 
                className="predict-button"
                onClick={handleTrain}
                disabled={loading}
              >
                {loading && activeAction === 'train' ? 'Training...' : 'Train Model'}
              </button>
            </div>
              
            {activeAction === 'train' && loading && (
              <div className="progress-container">
                <div className="progress-bar-container">
                  <div 
                    className="progress-bar"
                    style={{ width: `${progress}%` }}
                  />
                  <span>{progress}%</span>
                </div>
              </div>
            )}
            
            {activeAction === 'train' && !loading && progress === 100 && (
              <div className="success-message">
                Training completed successfully!
              </div>
            )}

            <div className="tft-action-block">
              <button 
                className="predict-button"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading && activeAction === 'predict' ? 'Predicting...' : 'Predict'}
              </button>
            </div>
            
            {activeAction === 'predict' && prediction && (
              <div className="prediction-result">
                <h3>Predicted Price:</h3>
                <p className="predicted-value">₹{prediction.toFixed(2)}</p>
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TFTModel;
