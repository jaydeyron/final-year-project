
#!/usr/bin/env python3
import os
import sys
import subprocess
import time

def retrain_models(symbols=None, start_date="2018-01-01", epochs=100):
    """Retrain models with more data and varied parameters"""
    
    # Use all symbols if none specified
    if symbols is None:
        symbols = ["RELIANCE", "TCS", "SENSEX", "INFY", "HDFCBANK"]
    
    print(f"Starting retraining of {len(symbols)} models...")
    
    for symbol in symbols:
        print(f"\n{'='*60}\nRetraining model for {symbol}\n{'='*60}")
        
        cmd = [
            sys.executable,
            'train.py',
            '--symbol', f"{symbol}.BO" if symbol != "SENSEX" else "^BSESN",
            '--display-symbol', symbol,
            '--start-date', start_date,
            '--epochs', str(epochs)
        ]
        
        # Run training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline().decode('utf-8')
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            print(f"Error retraining {symbol}: {rc}")
            error = process.stderr.read().decode('utf-8')
            print(f"Error details: {error}")
        
        # Wait a bit between models
        time.sleep(1)
    
    print("\nRetraining completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrain multiple models with improved settings')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to retrain')
    parser.add_argument('--start-date', type=str, default="2018-01-01", help='Start date for training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for each model')
    
    args = parser.parse_args()
    symbols = args.symbols.split(',') if args.symbols else None
    
    retrain_models(symbols, args.start_date, args.epochs)