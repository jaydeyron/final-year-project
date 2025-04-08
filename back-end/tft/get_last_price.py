import argparse
import sys
import os
import yfinance as yf
from datetime import datetime, timedelta

# Use absolute imports with the full path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both directories to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def get_last_closing_price(symbol, days_to_try=15):
    """
    Get the most recent closing price for a stock symbol
    
    Args:
        symbol: Stock symbol (e.g., "TCS.BO")
        days_to_try: Number of days to look back if today's data isn't available
        
    Returns:
        The last closing price as a float
    """
    try:
        # Get data for the past few days to ensure we have the latest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_try)
        
        # Format dates
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Fetch the data silently
        data = yf.download(
            symbol, 
            start=start_date_str, 
            end=end_date_str, 
            progress=False
        )
        
        # Check if data was retrieved
        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Get the last available closing price
        last_close = float(data['Close'].iloc[-1])
        
        return last_close
        
    except Exception as e:
        print(f"Error fetching closing price for {symbol}: {str(e)}", file=sys.stderr)
        raise e

def main():
    parser = argparse.ArgumentParser(description='Get last closing price for a stock symbol')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., "TCS.BO")')
    args = parser.parse_args()
    
    try:
        # Get the last closing price
        last_close = get_last_closing_price(args.symbol)
        
        # Print only the price to stdout for easy parsing by other scripts
        print(f"{last_close:.2f}")
        
    except Exception as e:
        # Print error to stderr so it doesn't interfere with the price output
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
