import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the stocks with consistent early start date
stocks = [
    {"name": "SENSEX", "symbol": "^BSESN", "startDate": "1957-07-01"},
    {"name": "Asian Paints", "symbol": "ASIANPAINT.BO", "startDate": "1957-07-01"},
    {"name": "Axis Bank", "symbol": "AXISBANK.BO", "startDate": "1957-07-01"},
    {"name": "Bajaj Finance", "symbol": "BAJFINANCE.BO", "startDate": "1957-07-01"},
    {"name": "Bharti Airtel", "symbol": "BHARTIARTL.BO", "startDate": "1957-07-01"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK.BO", "startDate": "1957-07-01"},
    {"name": "HUL", "symbol": "HINDUNILVR.BO", "startDate": "1957-07-01"},
    {"name": "ICICI Bank", "symbol": "ICICIBANK.BO", "startDate": "1957-07-01"},
    {"name": "Infosys", "symbol": "INFY.BO", "startDate": "1957-07-01"},
    {"name": "ITC", "symbol": "ITC.BO", "startDate": "1957-07-01"},
    {"name": "Kotak Bank", "symbol": "KOTAKBANK.BO", "startDate": "1957-07-01"},
    {"name": "L&T", "symbol": "LT.BO", "startDate": "1957-07-01"},
    {"name": "Maruti Suzuki", "symbol": "MARUTI.BO", "startDate": "1957-07-01"},
    {"name": "Power Grid", "symbol": "POWERGRID.BO", "startDate": "1957-07-01"},
    {"name": "Reliance Industries", "symbol": "RELIANCE.BO", "startDate": "1957-07-01"},
    {"name": "SBI", "symbol": "SBIN.BO", "startDate": "1957-07-01"},
    {"name": "Sun Pharma", "symbol": "SUNPHARMA.BO", "startDate": "1957-07-01"},
    {"name": "Tata Motors", "symbol": "TATAMOTORS.BO", "startDate": "1957-07-01"},
    {"name": "Tata Steel", "symbol": "TATASTEEL.BO", "startDate": "1957-07-01"},
    {"name": "TCS", "symbol": "TCS.BO", "startDate": "1957-07-01"}
]

def verify_start_date(stock):
    try:
        # Add .BO suffix if not SENSEX
        ticker = yf.Ticker(stock["symbol"])
        
        # Get data from start date
        start_date = datetime.strptime(stock["startDate"], "%Y-%m-%d")
        end_date = start_date + timedelta(days=7)  # Get a week's worth of data
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if (hist.empty):
            print(f"\n{stock['name']} ({stock['symbol']}):")
            print(f"No data available for start date: {stock['startDate']}")
            
            # Try to get the earliest available data
            hist = ticker.history(period="max")
            if not hist.empty:
                earliest_date = hist.index[0].strftime("%Y-%m-%d")
                print(f"Earliest available data: {earliest_date}")
                print(f"First record: \n{hist.iloc[0]}")
        else:
            print(f"\n{stock['name']} ({stock['symbol']}):")
            print(f"Data available from start date: {stock['startDate']}")
            print(f"First record: \n{hist.iloc[0]}")
            
    except Exception as e:
        print(f"\n{stock['name']} ({stock['symbol']}):")
        print(f"Error: {str(e)}")

def main():
    print("Verifying start dates for all stocks...")
    for stock in stocks:
        verify_start_date(stock)

if __name__ == "__main__":
    main()
