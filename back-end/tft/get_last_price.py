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
from tft.utils.data_loader import fetch_data

def main():
    parser = argparse.ArgumentParser(description='Get the last closing price for a stock')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    args = parser.parse_args()
    
    try:
        # Fetch just the latest few days of data
        data = fetch_data(args.symbol, days=5)
        
        if data is None or len(data) == 0:
            print(f"Error: No data available for {args.symbol}", file=sys.stderr)
            sys.exit(1)
            
        # Print only the last closing price
        last_close = data['Close'].iloc[-1]
        print(f"{last_close:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
