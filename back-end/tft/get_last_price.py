import argparse
import sys
import os
from datetime import datetime, timedelta
from utils.data_loader import fetch_data
from config import Config

def main():
    parser = argparse.ArgumentParser(description='Get last closing price for a symbol')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol for data fetching')
    args = parser.parse_args()
    
    # Suppress all warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        # Get end date as today and start date a week ago
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Redirect stderr to avoid Yahoo Finance warnings
        stderr_fd = sys.stderr.fileno()
        with os.fdopen(os.dup(stderr_fd), 'wb') as copied_stderr:
            sys.stderr = open(os.devnull, 'w')
            
            try:
                # Fetch recent data for the symbol
                recent_data = fetch_data(args.symbol, start_date, end_date, days=7)
                
                # Get the last close price
                if recent_data is not None and len(recent_data) > 0:
                    last_close = recent_data[-1][3]  # Close price is at index 3
                    print(f"{last_close:.2f}")
                else:
                    print("0.00")  # Fallback if no data is available
                    
            finally:
                # Restore stderr
                sys.stderr = os.fdopen(stderr_fd, 'w')
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print("0.00")  # Print default on error
        sys.exit(1)

if __name__ == "__main__":
    main()
