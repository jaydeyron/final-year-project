import argparse
import os
import sys
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Use absolute imports with the full path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both directories to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary modules
from tft.utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from tft.utils.model_utils import load_model
from tft.config import Config

def evaluate_model(symbol, test_days=30, plots_dir=None, ensemble=False):
    """
    Evaluate model performance on recent data and generate performance metrics
    
    Args:
        symbol: Stock symbol to evaluate
        test_days: Number of days to use for testing
        plots_dir: Directory to save plots (default: backend/plots/<symbol>)
        ensemble: Whether to use ensemble prediction (multiple sequences)
        
    Returns:
        Dictionary of performance metrics
    """
    # Clean symbol for model loading
    clean_symbol = symbol
    if symbol.endswith('.BO'):
        clean_symbol = symbol[:-3]
    elif symbol.startswith('^'):
        clean_symbol = symbol[1:]
    
    # Create plots directory if specified or use default
    if plots_dir is None:
        plots_dir = os.path.join(parent_dir, 'plots', clean_symbol)
    
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots will be saved to: {plots_dir}")
    
    # Load model and metadata
    print(f"Loading model for {clean_symbol}...")
    model, scaler_params, metadata = load_model(clean_symbol)
    
    if model is None:
        print(f"No model found for {clean_symbol}")
        return None
    
    # Get data for evaluation
    print(f"Fetching {test_days} days of data for {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=test_days + 60)  # Get extra data for proper sequences
    
    data = fetch_data(symbol, start_date=start_date.strftime('%Y-%m-%d'), 
                     end_date=end_date.strftime('%Y-%m-%d'))
    
    if data is None or len(data) < Config.SEQ_LENGTH + 5:
        print(f"Insufficient data for evaluation. Need at least {Config.SEQ_LENGTH + 5} data points.")
        return None
    
    # Set model to evaluation mode
    model.to(Config.DEVICE)
    model.eval()
    
    # Track actual and predicted values
    actuals = []
    predictions = []
    dates = []
    
    # Process data in evaluation window
    processed_data, _ = preprocess_data(data, scaler_params)
    
    # Determine start of evaluation period
    eval_window = min(len(data) - Config.SEQ_LENGTH, test_days)
    
    print(f"Evaluating on {eval_window} days...")
    
    # Make predictions for each day in the evaluation window
    for i in range(eval_window):
        # Use a sliding window approach
        seq_start = len(processed_data) - eval_window - Config.SEQ_LENGTH + i
        seq_end = seq_start + Config.SEQ_LENGTH
        
        if seq_start < 0 or seq_end > len(processed_data):
            continue
            
        # Extract sequence
        sequence = torch.FloatTensor(processed_data[seq_start:seq_end]).unsqueeze(0).to(Config.DEVICE)
        
        # Get actual price for the next day (or current day if at the end)
        target_idx = min(seq_end, len(data) - 1)
        actual_price = data['Close'].iloc[target_idx]
        target_date = data.index[target_idx]
        
        # Store actual price and date
        actuals.append(actual_price)
        dates.append(target_date)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence, apply_constraint=False).item()
        
        # Convert prediction to price
        price_idx = 3  # Index for Close price in the features
        price_mean = float(scaler_params['mean_'][price_idx])
        price_scale = float(scaler_params['scale_'][price_idx])
        
        # Scale back to price
        predicted_price = prediction * price_scale + price_mean
        predictions.append(predicted_price)
    
    # Calculate metrics
    metrics = {}
    
    # Convert to numpy arrays
    actuals_np = np.array(actuals)
    predictions_np = np.array(predictions)
    
    # Basic error metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(actuals_np, predictions_np))
    metrics['mae'] = mean_absolute_error(actuals_np, predictions_np)
    metrics['mape'] = np.mean(np.abs((actuals_np - predictions_np) / actuals_np)) * 100
    metrics['r2'] = r2_score(actuals_np, predictions_np)
    
    # Directional accuracy
    price_changes_actual = np.diff(actuals_np)
    price_changes_pred = np.diff(predictions_np)
    correct_directions = np.sum((price_changes_actual * price_changes_pred) > 0)
    metrics['directional_accuracy'] = correct_directions / len(price_changes_actual) * 100
    
    # Calculate trading metrics
    signals = []
    current_position = None
    trades = 0
    profitable_trades = 0
    
    for i in range(1, len(actuals_np)):
        # Generate simple trading signals
        if predictions_np[i] > actuals_np[i-1] * 1.01:  # 1% threshold for buy signal
            signal = 'buy'
        elif predictions_np[i] < actuals_np[i-1] * 0.99:  # 1% threshold for sell signal
            signal = 'sell'
        else:
            signal = 'hold'
        
        signals.append(signal)
        
        # Track position changes
        if current_position is None and signal in ['buy', 'sell']:
            current_position = signal
        elif current_position == 'buy' and signal == 'sell':
            trades += 1
            if actuals_np[i] > actuals_np[i-1]:  # Check if trade was profitable
                profitable_trades += 1
            current_position = 'sell'
        elif current_position == 'sell' and signal == 'buy':
            trades += 1
            if actuals_np[i] < actuals_np[i-1]:  # Short positions profit from price drops
                profitable_trades += 1
            current_position = 'buy'
    
    # Trading performance metrics
    metrics['total_trades'] = trades
    metrics['profitable_trades'] = profitable_trades
    metrics['profitable_trade_ratio'] = (profitable_trades / trades) * 100 if trades > 0 else 0
    
    # Print metrics summary
    print("\nPerformance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Profitable Trades: {metrics['profitable_trades']} ({metrics['profitable_trade_ratio']:.2f}%)")
    
    # Create DataFrame for visualization
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Predicted': predictions
    }).set_index('Date')
    
    # Generate plots
    
    # 1. Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Actual'], label='Actual Price', color='blue')
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted Price', color='red', linestyle='--')
    plt.title(f'{clean_symbol} - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    
    # 2. Prediction Error Over Time
    plt.figure(figsize=(12, 6))
    error = results_df['Predicted'] - results_df['Actual']
    plt.bar(results_df.index, error, color='green', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f'{clean_symbol} - Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error (Predicted - Actual)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_error.png'))
    
    # 3. Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(error, kde=True, color='purple')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{clean_symbol} - Prediction Error Distribution')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
    
    # 4. Predicted vs Actual Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
    plt.plot([results_df['Actual'].min(), results_df['Actual'].max()], 
             [results_df['Actual'].min(), results_df['Actual'].max()], 
             'r--')
    plt.title(f'{clean_symbol} - Predicted vs Actual Price')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_scatter.png'))
    
    # 5. Trading Signals Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot prices
    plt.subplot(2, 1, 1)
    plt.plot(results_df.index[1:], actuals_np[1:], label='Actual Price', color='blue')
    plt.plot(results_df.index[1:], predictions_np[1:], label='Predicted Price', color='red', linestyle='--')
    
    # Add buy/sell markers
    buy_signals = [i for i, signal in enumerate(signals) if signal == 'buy']
    sell_signals = [i for i, signal in enumerate(signals) if signal == 'sell']
    
    plt.scatter(results_df.index[1:][buy_signals], actuals_np[1:][buy_signals], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(results_df.index[1:][sell_signals], actuals_np[1:][sell_signals], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{clean_symbol} - Trading Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot returns
    plt.subplot(2, 1, 2)
    returns = np.diff(actuals_np) / actuals_np[:-1] * 100
    plt.bar(results_df.index[1:-1], returns, color='blue', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Daily Returns (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'trading_signals.png'))
    
    # Save metrics to JSON
    metrics['evaluation_date'] = datetime.now().strftime('%Y-%m-%d')
    metrics['symbol'] = clean_symbol
    metrics['data_points'] = len(actuals)
    metrics['test_period'] = f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    
    with open(os.path.join(plots_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nAll metrics and plots saved to {plots_dir}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate TFT model performance')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., "TCS.BO")')
    parser.add_argument('--days', type=int, default=30, help='Number of days to use for testing')
    parser.add_argument('--output-dir', type=str, help='Directory to save plots')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble prediction')
    args = parser.parse_args()
    
    evaluate_model(
        symbol=args.symbol,
        test_days=args.days,
        plots_dir=args.output_dir,
        ensemble=args.ensemble
    )

if __name__ == "__main__":
    main()
