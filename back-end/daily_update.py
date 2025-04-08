import os
import sys
import subprocess
import logging
import json
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='daily_update.log'
)
logger = logging.getLogger(__name__)

# Path to the models directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(backend_dir, 'models')

def main():
    logger.info("Starting daily model updates")
    
    # Get list of all model directories
    try:
        symbols = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        logger.info(f"Found {len(symbols)} models to update")
    except Exception as e:
        logger.error(f"Error listing model directories: {str(e)}")
        return
    
    for symbol in symbols:
        try:
            # Check if metadata exists to determine if this is a valid model
            metadata_path = os.path.join(models_dir, symbol, 'metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning(f"Skipping {symbol}: No metadata.json found")
                continue
                
            # Read metadata to get the BSE symbol
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            bse_symbol = metadata.get('bse_symbol')
            if not bse_symbol:
                logger.warning(f"Skipping {symbol}: No BSE symbol in metadata")
                continue
                
            logger.info(f"Updating model for {symbol} (BSE: {bse_symbol})")
            
            # Run the incremental update script
            cmd = [
                sys.executable,
                'tft/incremental_update.py',
                '--symbol', bse_symbol,
                '--display-symbol', symbol,
                '--epochs', '5'  # Use fewer epochs for daily updates
            ]
            
            # Execute update
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully updated {symbol}")
            else:
                logger.error(f"Error updating {symbol}: {result.stderr}")
                
            # Add small delay between updates
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Exception updating {symbol}: {str(e)}")
            continue
    
    logger.info("Daily update process completed")

if __name__ == "__main__":
    main()
