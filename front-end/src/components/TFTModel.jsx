import React, { useState, useEffect } from 'react';
import { niftyStocks } from '../data/niftyStocks';

function TFTModel({ symbol }) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [activeAction, setActiveAction] = useState(null); // 'train' or 'predict'
  const [useCustomDate, setUseCustomDate] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [lastClose, setLastClose] = useState(null);
  const [recommendation, setRecommendation] = useState('');
  const [priceChange, setPriceChange] = useState(null);
  const [strikeOptions, setStrikeOptions] = useState({});

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

  // Get last closing price
  const fetchLastClose = async (symbol) => {
    try {
      const actualSymbol = getActualSymbol(symbol);
      // This would ideally be a proper API call to get the latest closing price
      // For now, we'll use a simple approach to fetch this from our prediction API
      
      const response = await fetch(`${API_BASE_URL}/last-close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: actualSymbol })
      });
      
      if (response.ok) {
        const data = await response.json();
        setLastClose(data.lastClose);
        return data.lastClose;
      }
      
      return null;
    } catch (error) {
      console.error('Error fetching last close price:', error);
      return null;
    }
  };

  // Calculate recommended strike prices
  const calculateStrikePrices = (currentPrice, predictedPrice) => {
    if (!currentPrice || !predictedPrice) return null;
    
    const roundToNearestFive = (num) => {
      return Math.round(num / 5) * 5;
    };
    
    const isPriceUp = predictedPrice > currentPrice;
    const change = Math.abs(predictedPrice - currentPrice);
    const percentChange = (change / currentPrice) * 100;
    
    // Base strike prices on current price with adjustments
    let ceStrike, peStrike;
    
    if (isPriceUp) {
      // For upward movement, CE strike is slightly below current price
      ceStrike = roundToNearestFive(currentPrice * 0.99);
      // PE strike is higher than predicted price
      peStrike = roundToNearestFive(predictedPrice * 1.01);
    } else {
      // For downward movement, PE strike is slightly above current price
      peStrike = roundToNearestFive(currentPrice * 1.01);
      // CE strike is lower than predicted price
      ceStrike = roundToNearestFive(predictedPrice * 0.99);
    }
    
    return {
      callStrike: ceStrike,
      putStrike: peStrike,
      direction: isPriceUp ? 'bullish' : 'bearish',
      strength: percentChange > 1 ? 'strong' : 'mild'
    };
  };

  // Generate a recommendation based on predicted vs. current price
  const generateRecommendation = (predicted, current) => {
    if (!predicted || !current) return '';
    
    const change = predicted - current;
    const percentChange = (change / current) * 100;
    setPriceChange({
      absolute: change.toFixed(2),
      percent: percentChange.toFixed(2)
    });
    
    // Calculate strike prices
    const strikes = calculateStrikePrices(current, predicted);
    setStrikeOptions(strikes);
    
    if (percentChange > 1) {
      return "Strong bullish outlook. Consider buying Call Option (CE) or selling Put Option (PE).";
    } else if (percentChange > 0) {
      return "Slightly bullish. Consider buying Call Option (CE) with caution.";
    } else if (percentChange < -1) {
      return "Strong bearish outlook. Consider buying Put Option (PE) or selling Call Option (CE).";
    } else {
      return "Slightly bearish. Consider buying Put Option (PE) with caution.";
    }
  };

  const handleTrain = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);
      setActiveAction('train');
      setProgress(0); // Reset progress

      const actualSymbol = getActualSymbol(symbol);
      console.log('Starting training for symbol:', actualSymbol);
      
      // Prepare request body with optional start date
      const requestBody = { 
        symbol: actualSymbol
      };
      
      if (useCustomDate && startDate) {
        // Ensure date is in YYYY-MM-DD format
        const dateObject = new Date(startDate);
        const formattedDate = dateObject.toISOString().split('T')[0];
        
        // Double-check format is valid
        if (/^\d{4}-\d{2}-\d{2}$/.test(formattedDate)) {
          requestBody.start_date = formattedDate;
          console.log(`Using custom start date: ${formattedDate}`);
        } else {
          console.error(`Invalid date format: ${startDate} → ${formattedDate}`);
          setError("Invalid date format. Please use YYYY-MM-DD format.");
          setLoading(false);
          return;
        }
      }
      
      // Send training request
      console.log(`Sending training request to ${API_BASE_URL}/train`);
      const trainResponse = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      console.log('Training response status:', trainResponse.status);

      if (!trainResponse.ok) {
        // Handle error response
        try {
          const contentType = trainResponse.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorData = await trainResponse.json();
            throw new Error(errorData.detail || 'Training failed');
          } else {
            const errorText = await trainResponse.text();
            console.error('Non-JSON error response:', errorText);
            throw new Error(`Training failed with status: ${trainResponse.status}`);
          }
        } catch (parseError) {
          throw new Error(`Training failed: ${parseError.message || 'Unknown error'}`);
        }
      }

      // Start progress polling with improved error handling
      console.log('Starting progress polling...');
      let lastProgress = 0;
      let stagnantCount = 0;
      
      const progressInterval = setInterval(async () => {
        try {
          // Add timestamp to prevent caching - use a consistent API endpoint with complete URL
          const timestamp = new Date().getTime();
          const progressUrl = `http://localhost:8000/progress?_=${timestamp}`;
          
          console.log(`Polling progress at ${new Date().toISOString()}`);
          
          const progressResponse = await fetch(progressUrl, {
            headers: {
              'Cache-Control': 'no-cache, no-store, must-revalidate',
              'Pragma': 'no-cache',
              'Accept': 'application/json'
            }
          });
          
          // Get the response content type
          const contentType = progressResponse.headers.get('content-type');
          
          // Check if response is not OK
          if (!progressResponse.ok) {
            console.error(`Progress check failed: ${progressResponse.status}`);
            stagnantCount++;
            return;
          }
          
          // Parse response based on content type
          let progressData = null;
          
          // Only try to parse as JSON if the content type is json
          if (contentType && contentType.includes('application/json')) {
            const responseText = await progressResponse.text();
            console.log('Progress response:', responseText);
            
            try {
              progressData = JSON.parse(responseText);
            } catch (jsonError) {
              console.error('Error parsing JSON progress:', jsonError);
              console.log('Invalid JSON response:', responseText);
              throw new Error('Invalid JSON response from server');
            }
          } else {
            const textResponse = await progressResponse.text();
            console.warn('Unexpected content type:', contentType);
            console.log('Raw response:', textResponse);
            
            // Try to extract progress from text
            try {
              // Look for something like: {"progress": 45}
              const match = textResponse.match(/["']?progress["']?\s*:\s*(\d+)/i);
              if (match && match[1]) {
                progressData = { progress: parseInt(match[1]) };
                console.log('Extracted progress from text:', progressData);
              } else {
                throw new Error('Could not extract progress value from response');
              }
            } catch (extractError) {
              console.error('Failed to extract progress:', extractError);
              throw new Error('Could not parse progress data');
            }
          }
          
          // Validate and update progress
          if (progressData && typeof progressData.progress !== 'undefined') {
            // Convert string progress to number if needed
            const progressValue = typeof progressData.progress === 'string'
              ? parseInt(progressData.progress)
              : progressData.progress;
              
            // Only update if we got a valid number
            if (!isNaN(progressValue)) {
              console.log(`Progress update: ${progressValue}%`);
              setProgress(progressValue);
              
              // Handle stagnation detection
              if (progressValue === lastProgress) {
                stagnantCount++;
                console.log(`Progress stagnant at ${progressValue}% for ${stagnantCount} checks`);
                
                if (stagnantCount > 15 && progressValue < 100) {
                  console.warn('Progress appears to be stuck');
                }
              } else {
                stagnantCount = 0;
                lastProgress = progressValue;
              }
              
              // Error and completion detection
              if (progressValue === -1 || progressData.error) {
                console.error('Training error:', progressData.error);
                clearInterval(progressInterval);
                setLoading(false);
                setError(progressData.error || 'Training failed');
                return;
              }
              
              if (progressValue >= 100) {
                console.log('Training completed');
                clearInterval(progressInterval);
                setLoading(false);
              }
            } else {
              console.warn('Invalid progress value:', progressData.progress);
              stagnantCount++;
            }
          } else {
            console.warn('Missing progress data in response');
            stagnantCount++;
          }
        } catch (err) {
          console.error('Error checking progress:', err);
          stagnantCount++;
          
          if (stagnantCount > 5) {
            clearInterval(progressInterval);
            setLoading(false);
            setError(`Failed to check training progress: ${err.message}`);
          }
        }
      }, 2000); // Poll every 2 seconds

      // Add a safety timeout to stop polling after 30 minutes
      setTimeout(() => {
        if (progressInterval) {
          clearInterval(progressInterval);
          if (loading) {
            setLoading(false);
            setError('Training timed out. The model may still be training in the background.');
          }
        }
      }, 30 * 60 * 1000); // 30 minutes

    } catch (err) {
      console.error('Error in handleTrain:', err);
      setError(err.message || 'Unknown training error');
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);
      setRecommendation('');
      setPriceChange(null);
      setActiveAction('predict');

      const actualSymbol = getActualSymbol(symbol);
      console.log('Starting prediction for symbol:', actualSymbol);
      
      // Fetch last closing price first
      let closePrice = null;
      try {
        const closeResponse = await fetch(`${API_BASE_URL}/last-close`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: actualSymbol })
        });
        
        // Check if response is JSON before parsing
        const contentType = closeResponse.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const closeData = await closeResponse.json();
          closePrice = closeData.lastClose;
        } else {
          console.warn('Invalid JSON response from last-close endpoint');
        }
      } catch (closeErr) {
        console.error('Error fetching last close:', closeErr);
        // Continue with prediction even if last price fetch fails
      }
      
      // Send prediction request
      const predictResponse = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: actualSymbol })
      });
      
      if (!predictResponse.ok) {
        // Check content type before trying to parse as JSON
        const contentType = predictResponse.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await predictResponse.json();
          throw new Error(errorData.detail || 'Prediction failed');
        } else {
          // If not JSON, get text response
          const errorText = await predictResponse.text();
          console.error('Non-JSON error response:', errorText);
          throw new Error(`Prediction failed with status: ${predictResponse.status}`);
        }
      }
      
      // Check content type before parsing
      const contentType = predictResponse.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const responseText = await predictResponse.text();
        console.error('Unexpected non-JSON response:', responseText);
        throw new Error('Server returned invalid format. Expected JSON.');
      }
      
      const result = await predictResponse.json();
      console.log('Received prediction:', result);
      
      if (typeof result.prediction !== 'number') {
        throw new Error(`Invalid prediction value: ${result.prediction}`);
      }
      
      setPrediction(result.prediction);
      
      // For demo purposes, if we couldn't get last close from API, use a mock value
      const currentPrice = closePrice || result.prediction * 0.98; // Fallback
      setLastClose(currentPrice);
      
      // Generate recommendation
      const rec = generateRecommendation(result.prediction, currentPrice);
      setRecommendation(rec);
      
      setLoading(false);
    } catch (err) {
      console.error('Error in handlePredict:', err);
      setError(err.message || 'Failed to make prediction');
      setLoading(false);
    }
  };

  return (
    <div className="tradingview-widget-container">
      <div className="tft-widget dark-theme">
        <h2 className="widget-header">TFT Model Predictions</h2>
        <div className="widget-content">
          <div className="tft-actions-vertical">
            <div className="action-block">
              {/* Add custom date option */}
              <div className="custom-date-option">
                <label>
                  <input 
                    type="checkbox" 
                    checked={useCustomDate} 
                    onChange={e => setUseCustomDate(e.target.checked)} 
                  />
                  Use custom start date
                </label>
                {useCustomDate && (
                  <input 
                    type="date" 
                    value={startDate} 
                    onChange={e => setStartDate(e.target.value)}
                    className="date-input"
                  />
                )}
              </div>
              
              <button 
                className="action-button"
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

            <div className="action-block">
              <button 
                className="action-button"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading && activeAction === 'predict' ? 'Predicting...' : 'Predict'}
              </button>
            </div>
            
            {activeAction === 'predict' && prediction && (
              <div className="prediction-result">
                <div className="price-info">
                  <div className="price-row">
                    <span className="price-label">Current Price:</span>
                    <span className="current-price">{lastClose ? lastClose.toFixed(2) : 'N/A'}</span>
                  </div>
                  <div className="price-row">
                    <span className="price-label">Predicted Price:</span>
                    <span className={`predicted-value ${prediction > lastClose ? 'positive' : 'negative'}`}>
                      {prediction.toFixed(2)}
                    </span>
                  </div>
                  
                  {priceChange && (
                    <div className="price-change">
                      <span className={priceChange.absolute > 0 ? 'positive' : 'negative'}>
                        {priceChange.absolute > 0 ? '+' : ''}{priceChange.absolute} ({priceChange.absolute > 0 ? '+' : ''}{priceChange.percent}%)
                      </span>
                    </div>
                  )}
                </div>
                
                {recommendation && (
                  <div className="recommendation">
                    <h3>Recommendation</h3>
                    <p>{recommendation}</p>
                    
                    {strikeOptions && strikeOptions.direction && (
                      <div className="strike-suggestions">
                        <h4>Suggested Strikes:</h4>
                        <div className={`suggestion ${strikeOptions.direction === 'bullish' ? 'primary' : 'secondary'}`}>
                          <span className="option-type">Call Option (CE):</span>
                          <span className="strike-value">{strikeOptions.callStrike}</span>
                          {strikeOptions.direction === 'bullish' && <span className="action-tag buy-tag">Buy</span>}
                          {strikeOptions.direction === 'bearish' && <span className="action-tag sell-tag">Sell</span>}
                        </div>
                        
                        <div className={`suggestion ${strikeOptions.direction === 'bearish' ? 'primary' : 'secondary'}`}>
                          <span className="option-type">Put Option (PE):</span>
                          <span className="strike-value">{strikeOptions.putStrike}</span>
                          {strikeOptions.direction === 'bearish' && <span className="action-tag buy-tag">Buy</span>}
                          {strikeOptions.direction === 'bullish' && <span className="action-tag sell-tag">Sell</span>}
                        </div>
                      </div>
                    )}
                  </div>
                )}
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
      
      {/* Updated Dark theme CSS */}
      <style jsx>{`
        .dark-theme {
          background-color: #1e222d;
          border: 1px solid #363c4e;
          border-radius: 4px;
          padding: 16px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
          color: #d1d4dc;
          margin-bottom: 20px;
        }
        
        .widget-header {
          color: #ffffff;
          border-bottom: 1px solid #363c4e;
          padding-bottom: 12px;
          margin-top: 0;
          font-weight: 600;
          font-size: 18px;
          letter-spacing: 0.2px;
        }
        
        .widget-content {
          padding: 12px 0;
        }
        
        .action-button {
          background-color: #2962ff;
          color: white;
          border: none;
          padding: 8px 12px;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
          font-size: 14px;
          transition: background-color 0.2s;
          width: 100%;
          max-width: 180px;
          margin-top: 10px;
        }
        
        .action-button:hover {
          background-color: #1e56e8;
        }
        
        .action-button:active {
          background-color: #1c4fd6;
        }
        
        .action-button:disabled {
          background-color: #1e3a8a;
          cursor: not-allowed;
          opacity: 0.6;
        }
        
        .progress-bar {
          background-color: #2962ff;
          height: 100%;
          transition: width 0.3s ease;
        }
        
        .progress-bar-container {
          border: 1px solid #363c4e;
          border-radius: 4px;
          overflow: hidden;
          position: relative;
          height: 20px;
          margin: 12px 0;
          background-color: #2a2e39;
        }
        
        .progress-bar-container span {
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%);
          color: #ffffff;
          font-weight: 500;
          font-size: 12px;
          z-index: 1;
          text-shadow: 0 0 2px rgba(0, 0, 0, 0.5);
        }
        
        .predicted-value {
          font-weight: 700;
          font-size: 18px;
        }
        
        .custom-date-option {
          margin-bottom: 12px;
          color: #b2b5be;
          font-size: 14px;
        }
        
        .custom-date-option label {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        
        .date-input {
          margin-top: 8px;
          padding: 6px;
          border: 1px solid #363c4e;
          border-radius: 4px;
          width: 100%;
          color: #d1d4dc;
          background-color: #2a2e39;
          font-size: 14px;
        }
        
        .action-block {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin: 12px 0;
          width: 100%;
          background-color: #2a2e39;
          padding: 12px;
          border-radius: 4px;
          border: 1px solid #363c4e;
        }
        
        .success-message {
          color: #4bffb5;
          font-weight: 500;
          text-align: center;
          padding: 8px;
          background-color: rgba(75, 255, 181, 0.1);
          border-radius: 4px;
          margin: 12px 0;
          border-left: 3px solid #4bffb5;
          font-size: 14px;
        }
        
        .error-message {
          color: #ff5370;
          background-color: rgba(255, 83, 112, 0.1);
          padding: 8px;
          border-radius: 4px;
          margin-top: 12px;
          border-left: 3px solid #ff5370;
          font-size: 14px;
        }
        
        .price-info {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 16px;
          width: 100%;
        }
        
        .price-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 4px 10px;
          background-color: rgba(42, 46, 57, 0.5);
          border-radius: 4px;
        }
        
        .price-label {
          color: #b2b5be;
          font-size: 14px;
        }
        
        .current-price {
          font-size: 16px;
          font-weight: 500;
          color: #ffffff;
        }
        
        .price-change {
          text-align: right;
          margin-top: 4px;
          font-size: 14px;
          font-weight: 500;
        }
        
        .positive {
          color: #26a69a;
        }
        
        .negative {
          color: #ef5350;
        }
        
        .recommendation {
          margin-top: 20px;
          padding: 12px;
          background-color: rgba(42, 46, 57, 0.5);
          border-radius: 4px;
          border-left: 3px solid #2962ff;
        }
        
        .recommendation h3 {
          margin: 0 0 8px 0;
          font-size: 15px;
          color: #ffffff;
        }
        
        .recommendation p {
          margin: 0 0 15px 0;
          font-size: 13px;
          color: #d1d4dc;
          line-height: 1.5;
        }
        
        .strike-suggestions {
          margin-top: 15px;
          padding-top: 10px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .strike-suggestions h4 {
          margin: 0 0 10px 0;
          font-size: 14px;
          color: #b2b5be;
          font-weight: normal;
        }
        
        .suggestion {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 10px;
          margin-bottom: 6px;
          border-radius: 4px;
          background-color: rgba(42, 46, 57, 0.7);
        }
        
        .primary {
          border-left: 2px solid #26a69a;
        }
        
        .secondary {
          border-left: 2px solid #9e9e9e;
          opacity: 0.8;
        }
        
        .option-type {
          color: #d1d4dc;
          font-size: 13px;
          flex: 1;
        }
        
        .strike-value {
          color: #ffffff;
          font-weight: 600;
          font-size: 15px;
          margin: 0 10px;
        }
        
        .action-tag {
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }
        
        .buy-tag {
          background-color: rgba(38, 166, 154, 0.2);
          color: #26a69a;
        }
        
        .sell-tag {
          background-color: rgba(239, 83, 80, 0.2);
          color: #ef5350;
        }
        
        .prediction-result {
          border-top: 1px solid #363c4e;
          padding-top: 16px;
          margin-top: 16px;
          text-align: center;
        }
        
        input[type="checkbox"] {
          accent-color: #2962ff;
        }
      `}</style>
    </div>
  );
}

export default TFTModel;
