import argparse
import sys
from utils.data_loader import fetch_data
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Get last closing price for a stock')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., "TCS.BO")')
    args = parser.parse_args()
    
    try:
        # Fetch recent data using days parameter instead of date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch data using the days parameter
        data = fetch_data(
            args.symbol, 
            end_date=end_date,
            days=10  # Get last 10 days of data
        )
        
        if data is None or len(data) == 0:
            print(f"Error: No data found for {args.symbol}", file=sys.stderr)
            sys.exit(1)
            
        # Get the last close price
        last_close = data['Close'].iloc[-1]
        
        # Print only the number, formatted to 2 decimal places, no additional text
        print(f"{last_close:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
