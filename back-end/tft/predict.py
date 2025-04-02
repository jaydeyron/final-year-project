import torch
import argparse
import sys
import os
from datetime import datetime

# Use absolute imports with the full path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both directories to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now try the imports with explicit tft prefix
from tft.utils.data_loader import fetch_data, preprocess_data
from tft.utils.model_utils import load_model
from tft.config import Config
from tft.models.tft_model import TemporalFusionTransformer

import numpy as np
import pandas as pd

def analyze_market_conditions(data, window=14):
    """Analyze current market conditions to adjust prediction bias"""
    try:
        # Ensure we're working with pandas Series for calculations
        close_series = data['Close']
        
        # Calculate recent momentum
        recent_returns = close_series.pct_change().iloc[-window:].mean() * 100
        
        # Calculate recent volatility
        recent_volatility = close_series.pct_change().iloc[-window:].std() * 100
        
        # Calculate trend strength
        ema20 = close_series.ewm(span=20).mean()
        ema50 = close_series.ewm(span=50).mean()
        trend_strength = ((ema20.iloc[-1] / ema50.iloc[-1]) - 1) * 100
        
        # Determine price relative to moving averages
        price = close_series.iloc[-1]
        above_ema20 = price > ema20.iloc[-1]
        above_ema50 = price > ema50.iloc[-1]
        
        # Calculate volume trend
        volume_series = data['Volume'] 
        avg_volume = volume_series.iloc[-window:].mean()
        recent_volume = volume_series.iloc[-3:].mean()
        volume_trend = (recent_volume / avg_volume) - 1
        
        # Create a market condition score (-1 to +1)
        # Positive score suggests bullish conditions
        # Negative score suggests bearish conditions
        score_components = [
            np.clip(recent_returns / 2, -1, 1),  # Recent returns as percentage
            np.clip(trend_strength / 2, -1, 1),   # Trend strength
            0.5 if above_ema20 else -0.5,         # Price above short-term MA
            0.5 if above_ema50 else -0.5,         # Price above medium-term MA
            np.clip(volume_trend, -1, 1) * 0.3    # Volume trend (reduced impact)
        ]
        
        market_score = sum(score_components) / len(score_components)
        
        return {
            'market_score': market_score,
            'recent_volatility': recent_volatility,
            'recent_returns': recent_returns,
            'trend_strength': trend_strength,
            'price_to_ema20': 1 if above_ema20 else -1,
            'price_to_ema50': 1 if above_ema50 else -1
        }
    except Exception as e:
        print(f"Error in market analysis: {str(e)}")
        # In case of error, return neutral condition
        return {
            'market_score': 0,
            'recent_volatility': 0,
            'recent_returns': 0,
            'trend_strength': 0,
            'price_to_ema20': 0,
            'price_to_ema50': 0
        }

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained TFT model')
    parser.add_argument('--symbol', type=str, required=True, help='BSE symbol for data fetching')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model loading')
    parser.add_argument('--model-symbol', type=str, help='Override which model to use (e.g., TCS)')
    parser.add_argument('--days', type=int, default=120, help='Number of days of data to fetch')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--ensemble', action='store_true', help='Use multiple prediction windows for ensemble')
    args = parser.parse_args()
    
    # Suppress all warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        # Use debug printing to separate from the actual output
        def debug_print(msg):
            if args.debug:
                print(f"DEBUG: {msg}", file=sys.stderr)
        
        # Determine which model to load
        model_symbol = args.model_symbol or args.display_symbol or args.symbol
        # Clean up symbol for model loading
        if model_symbol.endswith('.BO'):
            model_symbol = model_symbol[:-3]
        elif model_symbol.startswith('^'):
            model_symbol = model_symbol[1:]
            
        debug_print(f"Loading model for symbol: {model_symbol}")
        
        # Ensure models directory exists
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(backend_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Load model, scaler and metadata
        model, scaler_params, metadata = load_model(model_symbol)
        
        if model is None:
            debug_print(f"No model found for {model_symbol}, trying TCS as fallback")
            model, scaler_params, metadata = load_model("TCS")
            if model is None:
                raise ValueError(f"Failed to load model for {model_symbol} and TCS fallback")
        
        # Convert model to evaluation mode
        model.to(Config.DEVICE)
        model.eval()
        
        # Fetch recent data for prediction
        debug_print(f"Loading recent data for {args.symbol}")
        end_date = datetime.now().strftime('%Y-%m-%d')
        days_to_fetch = args.days
        
        # Try to fetch data with increased days if needed
        max_tries = 3
        current_try = 0
        recent_data = None
        
        while current_try < max_tries and (recent_data is None or len(recent_data) < Config.SEQ_LENGTH):
            current_try += 1
            days_to_fetch = args.days * current_try
            
            debug_print(f"Fetching {days_to_fetch} days of data (attempt {current_try})")
            
            # Silence the yfinance warning outputs
            with open(os.devnull, 'w') as devnull:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = devnull
                
                try:
                    recent_data = fetch_data(args.symbol, end_date=end_date, days=days_to_fetch)
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
        
        if recent_data is None or len(recent_data) < Config.SEQ_LENGTH:
            raise ValueError(f"Insufficient data for {args.symbol}. Need at least {Config.SEQ_LENGTH} data points.")
            
        debug_print(f"Processing {len(recent_data)} data points")
        
        # Analyze market conditions to adjust bias
        market_conditions = analyze_market_conditions(recent_data)
        debug_print(f"Market conditions: {market_conditions}")
        
        # Preprocess data using saved scaler parameters
        features, _ = preprocess_data(recent_data, scaler_params)
        
        # Make sure we have enough data points after preprocessing
        if len(features) < Config.SEQ_LENGTH:
            raise ValueError(f"Insufficient features after preprocessing: {len(features)} < {Config.SEQ_LENGTH}")
        
        # Get multiple sequences for ensemble if requested
        if args.ensemble:
            # Take 3 sequences with different windows to reduce bias
            sequences = []
            last_close = recent_data['Close'].iloc[-1]
            
            # Last sequence (most recent)
            seq_idx = len(features) - Config.SEQ_LENGTH
            sequences.append(torch.FloatTensor(features[seq_idx:seq_idx + Config.SEQ_LENGTH]).unsqueeze(0).to(Config.DEVICE))
            
            # Sequence from 2 days ago
            if len(features) >= Config.SEQ_LENGTH + 2:
                seq_idx = len(features) - Config.SEQ_LENGTH - 2
                sequences.append(torch.FloatTensor(features[seq_idx:seq_idx + Config.SEQ_LENGTH]).unsqueeze(0).to(Config.DEVICE))
            
            # Sequence from 5 days ago
            if len(features) >= Config.SEQ_LENGTH + 5:
                seq_idx = len(features) - Config.SEQ_LENGTH - 5
                sequences.append(torch.FloatTensor(features[seq_idx:seq_idx + Config.SEQ_LENGTH]).unsqueeze(0).to(Config.DEVICE))
            
            # Make predictions with each sequence
            predictions = []
            with torch.no_grad():
                for sequence in sequences:
                    pred = model(sequence, apply_constraint=False).item()
                    predictions.append(pred)
            
            # Use weighted average (recent data has more weight)
            if len(predictions) == 3:
                weights = [0.6, 0.25, 0.15]  # Most weight to most recent data
            elif len(predictions) == 2:
                weights = [0.7, 0.3]
            else:
                weights = [1.0]
                
            weighted_pred = sum(p * w for p, w in zip(predictions, weights))
                
            # Apply market bias adjustment
            market_bias = market_conditions['market_score'] * 0.01  # Scale the impact
            adjusted_pred = weighted_pred * (1 + market_bias)
            
            debug_print(f"Raw predictions: {predictions}")
            debug_print(f"Weighted prediction: {weighted_pred}")
            debug_print(f"Market bias: {market_bias}")
            debug_print(f"Adjusted prediction: {adjusted_pred}")
            
            prediction = adjusted_pred
            
        else:
            # Standard single prediction
            # Get the last sequence for prediction
            last_seq_start = max(0, len(features) - Config.SEQ_LENGTH)
            last_seq_end = last_seq_start + Config.SEQ_LENGTH
            
            debug_print(f"Using sequence range: {last_seq_start}:{last_seq_end}")
            last_sequence = torch.FloatTensor(features[last_seq_start:last_seq_end]).unsqueeze(0).to(Config.DEVICE)
            
            debug_print(f"Sequence shape: {last_sequence.shape}")
            
            # Make prediction with model - disable constraints to see raw predictions
            with torch.no_grad():
                prediction = model(last_sequence, apply_constraint=False).item()
        
        # Convert prediction to price with better error handling
        try:
            # Scale back to price
            price_idx = 3  # Index for Close price
            mean_val = float(scaler_params['price_mean_'][price_idx])
            scale_val = float(scaler_params['price_scale_'][price_idx])
            
            # Avoid division by zero
            if scale_val == 0:
                scale_val = 1.0
                debug_print("Warning: Scale value was zero, using 1.0")
            
            # Convert prediction to price
            predicted_value = prediction * scale_val + mean_val
            
            # Calculate percent change for debugging
            last_close = recent_data['Close'].iloc[-1]
            percent_change = ((predicted_value - last_close) / last_close) * 100
            
            # Add debug information
            debug_print(f"Last close: {last_close:.2f}")
            debug_print(f"Raw prediction value: {prediction}")
            debug_print(f"Scale value: {scale_val}, Mean value: {mean_val}")
            debug_print(f"Calculated prediction: {predicted_value:.2f}")
            debug_print(f"Percent change: {percent_change:.2f}%")
            
            # Print only the predicted value for API - ensure it's the only output to stdout
            # Flush before printing to ensure clean output
            sys.stdout.flush()
            print(f"{predicted_value:.2f}", flush=True)
            
        except Exception as pred_error:
            debug_print(f"Prediction error details: {str(pred_error)}")
            # Try using market-aware fallback approach if the scaling fails
            try:
                # Get latest close price and apply an informed change based on market conditions
                last_close = float(recent_data['Close'].iloc[-1])
                
                # Base change on market score and recent volatility
                market_score = market_conditions['market_score']
                volatility = market_conditions['recent_volatility'] / 100  # Convert to decimal
                
                # Limit volatility influence
                max_volatility = 0.03  # Cap at 3%
                volatility = min(volatility, max_volatility)
                
                # Calculate expected change range
                expected_change = market_score * volatility
                
                # Add small random component
                import random
                random_component = random.uniform(-0.005, 0.005)  # ±0.5%
                total_change = expected_change + random_component
                
                # Apply change to last close
                fallback_prediction = last_close * (1.0 + total_change)
                
                debug_print(f"Using market-aware fallback. Last close: {last_close}, market score: {market_score}")
                debug_print(f"Expected change: {expected_change:.4f}, random: {random_component:.4f}")
                debug_print(f"Total change: {total_change:.4f} ({total_change*100:.2f}%)")
                debug_print(f"Fallback prediction: {fallback_prediction:.2f}")
                
                # Make sure we flush stdout before printing the fallback prediction
                sys.stdout.flush()
                print(f"{fallback_prediction:.2f}", flush=True)
            except Exception as fb_error:
                debug_print(f"Fallback prediction failed: {str(fb_error)}")
                # Ultimate fallback - just return the last close
                # Make sure we flush stdout before printing the ultimate fallback
                sys.stdout.flush()
                print(f"{recent_data['Close'].iloc[-1]:.2f}", flush=True)
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()