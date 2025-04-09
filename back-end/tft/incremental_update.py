# Fix the incremental update script with better error handling and debugging
import argparse
import os
import sys
import logging
import torch
import json
from datetime import datetime, timedelta
import numpy as np

# Use absolute imports with the full path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both directories to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import the required modules
from tft.utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from tft.utils.model_utils import save_model, load_model
from tft.config import Config
from tft.models.tft_model import TemporalFusionTransformer
from torch.utils.data import DataLoader

# Configure logging to also output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("incremental_update.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def update_progress(percent):
    """Update the progress.json file"""
    progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.json')
    try:
        with open(progress_file, 'w') as f:
            json.dump({"progress": percent, "update_type": "incremental"}, f)
    except Exception as e:
        logger.error(f"Error updating progress file: {str(e)}")

def update_metadata_only(symbol, metadata, reason):
    """Update metadata without modifying the model"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Create update info without changing model
        additional_info = {
            'data_start_date': metadata.get('data_start_date'),
            'data_end_date': today,  # Update to current date
            'symbol_name': metadata.get('symbol_name', symbol),
            'bse_symbol': metadata.get('bse_symbol', f"{symbol}.BO"),
            'training_completed': metadata.get('training_completed'),
            'epochs': metadata.get('epochs', 0),
            'data_points': metadata.get('data_points', 0),
            'last_update': datetime.now().isoformat(),
            'update_type': reason
        }
        
        # Save only the metadata file
        metadata_path = os.path.join(Config.MODEL_DIR, symbol, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(additional_info, f, indent=2)
            
        logger.info(f"Updated metadata only with reason: {reason}")
        
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Incrementally update TFT model with latest data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to update (e.g., "TCS.BO")')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model (e.g., "TCS")')
    parser.add_argument('--days', type=int, default=30, help='Number of days of new data to fetch if date not specified')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs for update')
    parser.add_argument('--start-date', type=str, help='Start date for incremental data (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Symbol to use for saving/loading the model
    save_symbol = args.display_symbol if args.display_symbol else args.symbol.replace('.BO', '')
    
    try:
        logger.info(f"Starting incremental update for {args.symbol}, model: {save_symbol}")
        update_progress(5)  # 5% - starting
        
        # Step 1: Load existing model and metadata
        logger.info(f"Loading model from directory: {Config.MODEL_DIR}/{save_symbol}")
        model_dir = os.path.join(Config.MODEL_DIR, save_symbol)
        if not os.path.exists(model_dir):
            logger.error(f"Model directory does not exist: {model_dir}")
            update_progress(-1)
            sys.exit(1)
            
        model, scaler_params, metadata = load_model(save_symbol)
        
        if model is None:
            logger.error(f"No existing model found for {save_symbol}. Please train a model first.")
            update_progress(-1)
            sys.exit(1)

        update_progress(10)  # 10% - model loaded
            
        # Get the date of last training from metadata or from argument
        if args.start_date:
            logger.info(f"Using provided start date: {args.start_date}")
            last_training_date = args.start_date
        elif metadata and metadata.get('data_end_date'):
            last_training_date = metadata.get('data_end_date')
            logger.info(f"Using last training date from metadata: {last_training_date}")
        else:
            # If no end date in metadata, use 30 days ago as default
            last_training_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            logger.warning(f"No previous training end date found. Using {last_training_date} as default.")
        
        # Step 2: Fetch new data
        today = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Fetching new data from {last_training_date} to {today}")
        update_progress(20)  # 20% - starting data fetch
        
        # Try multiple times with increasing date ranges if needed
        new_data = None
        # Convert last_training_date to datetime object
        try:
            last_date_obj = datetime.strptime(last_training_date, '%Y-%m-%d')
            
            # If last_training_date is today or in the future, adjust to yesterday
            if last_date_obj >= datetime.now():
                logger.warning("Last training date is today or in future, adjusting to get some data")
                last_date_obj = datetime.now() - timedelta(days=15)
                last_training_date = last_date_obj.strftime('%Y-%m-%d')
                
            # If last_training_date is too close to today, move back a bit to ensure we get some data
            days_diff = (datetime.now() - last_date_obj).days
            if days_diff < 3:
                logger.warning(f"Last training date too recent ({days_diff} days ago), moving back to get more data")
                last_date_obj = datetime.now() - timedelta(days=15)
                last_training_date = last_date_obj.strftime('%Y-%m-%d')
        except Exception as e:
            logger.error(f"Error parsing last training date: {e}")
            # Fall back to 15 days ago
            last_training_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
            
        # Try multiple date ranges
        attempts = 0
        while new_data is None or len(new_data) < 10:
            attempts += 1
            days_to_go_back = 15 * attempts
            
            # Adjust start date further back on each attempt
            adjusted_start = (datetime.strptime(last_training_date, '%Y-%m-%d') - 
                              timedelta(days=days_to_go_back)).strftime('%Y-%m-%d')
            
            logger.info(f"Attempt {attempts}: Fetching data from {adjusted_start} to {today}")
            new_data = fetch_data(args.symbol, start_date=adjusted_start, end_date=today)
            
            if attempts >= 3:
                break
        
        if new_data is None or len(new_data) < 5:  # Need at least some new data
            logger.warning(f"No substantial new data available for {args.symbol} since {last_training_date}")
            # Update metadata without updating model
            update_metadata_only(save_symbol, metadata, "no_new_data")
            update_progress(100)  # Complete
            return
            
        logger.info(f"Fetched {len(new_data)} new data points")
        update_progress(40)  # 40% - data fetched
        
        # Step 3: Process the new data using the existing scaler params
        logger.info("Preprocessing new data with existing scalers")
        
        # Check if scaler_params has the expected structure
        if not isinstance(scaler_params, dict) or 'mean_' not in scaler_params or 'scale_' not in scaler_params:
            logger.error("Invalid scaler parameters format")
            update_progress(-1)
            sys.exit(1)
            
        # Ensure arrays are numpy arrays
        if isinstance(scaler_params['mean_'], list):
            scaler_params['mean_'] = np.array(scaler_params['mean_'])
        if isinstance(scaler_params['scale_'], list):
            scaler_params['scale_'] = np.array(scaler_params['scale_'])
        
        # Add protection against zero scale values to avoid division by zero
        if np.any(scaler_params['scale_'] == 0):
            logger.warning("Found zero values in scaler parameters, fixing...")
            scaler_params['scale_'][scaler_params['scale_'] == 0] = 1.0
            
        scaled_data, scaler_params = preprocess_data(new_data, scaler_params)
        
        # Create training sequences from the new data
        logger.info("Creating training sequences")
        seq_length = Config.SEQ_LENGTH
        
        # Check if we have enough data for the sequence length
        if len(scaled_data) <= seq_length:
            logger.warning(f"Not enough data points ({len(scaled_data)}) for sequence length ({seq_length})")
            
            # If we don't have enough new data, just update metadata and complete
            if len(scaled_data) < 5:
                logger.info("Too few data points for incremental training, updating metadata only")
                update_metadata_only(save_symbol, metadata, "insufficient_data")
                update_progress(100)  # Complete
                return
                
            # Try with a smaller sequence length if possible
            if len(scaled_data) > 10:
                seq_length = len(scaled_data) - 5  # Leave some room
                logger.info(f"Adjusting sequence length to {seq_length}")
            else:
                logger.error("Cannot create sequences from the available data")
                update_progress(-1)
                sys.exit(1)
                
        xs, ys = create_sequences(scaled_data, seq_length)
        
        if len(xs) < 1:
            logger.warning("Not enough new data to create training sequences")
            update_metadata_only(save_symbol, metadata, "insufficient_sequences")
            update_progress(100)  # Complete
            return
            
        logger.info(f"Created {len(xs)} sequences from new data")
        update_progress(50)  # 50% - data processed
        
        # Step 4: Create dataset and dataloader
        train_dataset = TimeSeriesDataset(xs, ys)
        batch_size = min(8, len(xs))  # Use smaller batch size for fine-tuning
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Step 5: Fine-tune the model with new data
        logger.info(f"Fine-tuning model on {args.epochs} epochs")
        model.train()  # Set model to training mode
        model.to(Config.DEVICE)
        
        # Create optimizer with lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE * 0.1)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        for epoch in range(args.epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}')
            
            # Update progress based on epochs completed
            progress = 50 + int((epoch + 1) / args.epochs * 40)  # 50-90% during training
            update_progress(progress)
        
        # Step 6: Save updated model
        logger.info("Saving updated model")
        
        # Update metadata with new end date and additional training info
        total_epochs = metadata.get('epochs', 0) + args.epochs
        
        additional_info = {
            'data_start_date': metadata.get('data_start_date'),
            'data_end_date': today,  # Update to current date
            'symbol_name': metadata.get('symbol_name', save_symbol),
            'bse_symbol': args.symbol,
            'training_completed': metadata.get('training_completed'),
            'epochs': total_epochs,
            'data_points': metadata.get('data_points', 0) + len(new_data),
            'last_update': datetime.now().isoformat(),
            'update_type': 'incremental'
        }
        
        # Get input size from model or data
        input_size = xs.shape[2] if xs is not None else model.input_size
        
        # Save updated model
        saved_paths = save_model(
            save_symbol,
            model, 
            scaler_params, 
            input_size, 
            total_epochs, 
            additional_info
        )
        
        logger.info(f"Updated model saved to {saved_paths['model']}")
        logger.info(f"Incremental update completed successfully for {save_symbol}")
        update_progress(100)  # 100% - complete
        
    except Exception as e:
        logger.error(f"Error during incremental update: {str(e)}", exc_info=True)
        update_progress(-1)  # Error state
        sys.exit(1)

if __name__ == "__main__":
    main()
