import os
import torch
import json
import sys
import logging
import numpy as np
from datetime import datetime, timedelta  # Add this import

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.tft_model import TemporalFusionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model(symbol, model, scaler_params, input_size, epochs, additional_info=None):
    """
    Save model, scaler and metadata for a specific stock symbol
    
    Args:
        symbol: Stock symbol (for directory naming)
        model: Trained model
        scaler_params: Dictionary of scaler parameters
        input_size: Model input size
        epochs: Number of epochs trained
        additional_info: Dictionary of additional metadata
    
    Returns:
        Dictionary with paths to saved files
    """
    try:
        # Get paths for saving
        model_paths = Config.get_model_paths(symbol)
        model_path = model_paths['model']
        scaler_path = model_paths['scaler']
        metadata_path = model_paths['metadata']
        
        # Save the model - keeping the model on CPU
        model.cpu()
        torch.save(model.state_dict(), model_path)
        
        # Convert numpy arrays to lists for JSON serialization
        scaler_json = {}
        for key, value in scaler_params.items():
            if isinstance(value, np.ndarray):
                scaler_json[key] = value.tolist()
            else:
                scaler_json[key] = value
        
        # Save scaler parameters
        with open(scaler_path, 'w') as f:
            json.dump(scaler_json, f)
        
        # Create metadata
        metadata = {
            'symbol': symbol,
            'input_size': input_size,
            'hidden_size': model.hidden_size,
            'num_layers': Config.NUM_LAYERS,
            'num_heads': Config.NUM_HEADS,
            'dropout': Config.DROPOUT,
            'epochs': epochs,
            'creation_date': datetime.now().isoformat(),
        }
        
        # Add additional info if provided
        if additional_info:
            metadata.update(additional_info)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Return paths of saved files
        return {
            'model': model_path,
            'scaler': scaler_path,
            'metadata': metadata_path
        }
        
    except Exception as e:
        logger.error(f"Error saving model for {symbol}: {str(e)}")
        raise e  # Re-raise to handle in the calling function

# Update the adapt_old_model_to_new function to handle size mismatches better
def adapt_old_model_to_new(state_dict):
    """
    Adapt an old model state dict to be compatible with the new architecture
    
    Args:
        state_dict: The state dict loaded from an older model version
        
    Returns:
        A modified state dict compatible with the current architecture
    """
    logger.info("Adapting old model format to new architecture")
    new_state_dict = {}
    
    # Determine the hidden size from existing model
    hidden_size = 128  # Default size for older models
    if "lstm.weight_ih_l0" in state_dict:
        # Get dimensions from existing parameters
        hidden_size = state_dict["lstm.weight_ih_l0"].shape[0] // 4
        logger.info(f"Detected hidden size from model: {hidden_size}")
    
    # Initialize new state dict with adapted parameters
    for key, value in state_dict.items():
        # Handle renamed layers
        if key == "pre_output.weight":
            new_state_dict["pre_output_1.weight"] = value
        elif key == "pre_output.bias":
            new_state_dict["pre_output_1.bias"] = value
        else:
            # For LSTM layers, we need to handle dimension mismatches
            if key.startswith("lstm.weight_ih") and key.endswith(("l1", "l1_reverse", "l2", "l2_reverse")):
                # For these layers, we need to ensure they have the right dimension
                # The issue is that the input to layer 1 and 2 changed from 128 to 256
                # Handle this by padding with zeros
                value_shape = value.shape
                # Check if this is an input weight that needs resizing
                if "_ih_" in key and value_shape[1] == 128:
                    logger.info(f"Resizing {key} from {value_shape} to match current model architecture")
                    # Create a new tensor with the right shape, initialize to zeros
                    new_shape = (value_shape[0], 256)  # New input size is 256
                    new_tensor = torch.zeros(new_shape)
                    # Copy the old values into the first part
                    new_tensor[:, :value_shape[1]] = value
                    # Save the resized tensor
                    new_state_dict[key] = new_tensor
                else:
                    # Keep as-is for other weights
                    new_state_dict[key] = value
            else:
                # Keep all other parameters as-is
                new_state_dict[key] = value
    
    # Add missing bidirectional LSTM parameters if needed
    for layer in range(3):  # Assuming 3 LSTM layers
        # Check if reverse direction parameters are missing
        if f"lstm.weight_ih_l{layer}" in state_dict and f"lstm.weight_ih_l{layer}_reverse" not in state_dict:
            # Get shapes from forward direction
            ih_key = f"lstm.weight_ih_l{layer}"
            hh_key = f"lstm.weight_hh_l{layer}"
            
            if ih_key in state_dict and hh_key in state_dict:
                shape_ih = state_dict[ih_key].shape
                shape_hh = state_dict[hh_key].shape
                
                # For layers 1 and 2, ensure the input dimension is correct (256)
                if layer in (1, 2):
                    reverse_shape_ih = (shape_ih[0], 256)  # New input size is 256
                else:
                    reverse_shape_ih = shape_ih
                    
                # Create tensors of appropriate shape
                new_state_dict[f"lstm.weight_ih_l{layer}_reverse"] = torch.zeros(reverse_shape_ih)
                new_state_dict[f"lstm.weight_hh_l{layer}_reverse"] = torch.zeros(shape_hh)
                new_state_dict[f"lstm.bias_ih_l{layer}_reverse"] = torch.zeros(shape_ih[0])
                new_state_dict[f"lstm.bias_hh_l{layer}_reverse"] = torch.zeros(shape_ih[0])
    
    # Add LSTM reducer layer (for reducing bidirectional output)
    if "lstm_reducer.weight" not in state_dict:
        new_state_dict["lstm_reducer.weight"] = torch.eye(hidden_size).repeat(1, 2)
        new_state_dict["lstm_reducer.bias"] = torch.zeros(hidden_size)
    
    # Add temporal attention layers if missing
    if "temporal_attention.in_proj_weight" not in state_dict:
        new_state_dict["temporal_attention.in_proj_weight"] = torch.zeros(3 * hidden_size, hidden_size)
        new_state_dict["temporal_attention.in_proj_bias"] = torch.zeros(3 * hidden_size)
        new_state_dict["temporal_attention.out_proj.weight"] = torch.eye(hidden_size)
        new_state_dict["temporal_attention.out_proj.bias"] = torch.zeros(hidden_size)
    
    # Add new output layers
    if "pre_output_1.weight" not in state_dict:
        if "pre_output.weight" in state_dict:
            # Copy old pre_output to pre_output_1
            new_state_dict["pre_output_1.weight"] = state_dict["pre_output.weight"]
            new_state_dict["pre_output_1.bias"] = state_dict["pre_output.bias"]
        else:
            # Initialize if missing
            new_state_dict["pre_output_1.weight"] = torch.eye(hidden_size)
            new_state_dict["pre_output_1.bias"] = torch.zeros(hidden_size)
    
    if "pre_output_2.weight" not in state_dict:
        new_state_dict["pre_output_2.weight"] = torch.eye(hidden_size)
        new_state_dict["pre_output_2.bias"] = torch.zeros(hidden_size)
    
    # Add feature gate layers if missing
    if "feature_gate.0.weight" not in state_dict and "feature_layer.weight" in new_state_dict:
        input_size = new_state_dict["feature_layer.weight"].shape[1]
        new_state_dict["feature_gate.0.weight"] = torch.eye(input_size)
        new_state_dict["feature_gate.0.bias"] = torch.zeros(input_size)
    
    # Add uncertainty layer if missing
    if "uncertainty_layer.weight" not in state_dict and "output_layer.weight" in new_state_dict:
        output_size = new_state_dict["output_layer.weight"].shape[0]
        new_state_dict["uncertainty_layer.weight"] = torch.zeros(output_size, hidden_size)
        new_state_dict["uncertainty_layer.bias"] = torch.zeros(output_size)
    
    # Add layer normalization if missing
    for i in range(1, 4):
        norm_w_key = f"layer_norm{i}.weight"
        norm_b_key = f"layer_norm{i}.bias"
        if norm_w_key not in state_dict:
            new_state_dict[norm_w_key] = torch.ones(hidden_size)
            new_state_dict[norm_b_key] = torch.zeros(hidden_size)
    
    logger.info(f"Adaptation complete. Original keys: {len(state_dict)}, New keys: {len(new_state_dict)}")
    return new_state_dict

# Also update the load_model function to handle incompatible models
def load_model(symbol, device=None):
    """
    Load a trained model, scaler parameters and metadata for a given symbol
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        device: Optional torch device
        
    Returns:
        Tuple of (model, scaler_params, metadata)
    """
    try:
        # Get paths for model files
        model_paths = Config.get_model_paths(symbol)
        model_path = model_paths['model']
        scaler_path = model_paths['scaler']
        metadata_path = model_paths['metadata']
        
        # Check if all files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning(f"Model files not found for {symbol}")
            return None, None, None
        
        # First try getting metadata to determine the right model architecture
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Get input size from metadata if available
                input_size = metadata.get('input_size', Config.INPUT_SIZE)
                hidden_size = metadata.get('hidden_size', Config.HIDDEN_SIZE)
                logger.info(f"Using model parameters from metadata: input_size={input_size}, hidden_size={hidden_size}")
            except Exception as e:
                logger.error(f"Error reading metadata for {symbol}: {str(e)}")
                metadata = {}
                input_size = Config.INPUT_SIZE
                hidden_size = Config.HIDDEN_SIZE
        else:
            metadata = {}
            input_size = Config.INPUT_SIZE
            hidden_size = Config.HIDDEN_SIZE
            
        # If we can't load the model normally, try rebuilding it from scratch
        try:
            # First try loading with weights_only=True
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            
            # Initialize the model with parameters from metadata
            model = TemporalFusionTransformer(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=1,
                num_layers=Config.NUM_LAYERS,
                num_heads=Config.NUM_HEADS,
                dropout=Config.DROPOUT,
                max_change_percent=Config.MAX_CHANGE_PERCENT
            )
            
            # Check if this is an old model that needs adaptation
            if "pre_output.weight" in state_dict or "lstm.weight_ih_l0_reverse" not in state_dict:
                logger.info(f"Detected old model format for {symbol}, adapting...")
                state_dict = adapt_old_model_to_new(state_dict)
            
            # Try loading with strict=False to allow missing/extra parameters
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded model for {symbol} with non-strict loading")
            
            model.to(device)
            model.eval()  # Set model to evaluation mode
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            logger.info(f"Attempting to create a fresh model for {symbol}")
            
            # Create a new model if loading fails completely
            model = TemporalFusionTransformer(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=1,
                num_layers=Config.NUM_LAYERS,
                num_heads=Config.NUM_HEADS,
                dropout=Config.DROPOUT,
                max_change_percent=Config.MAX_CHANGE_PERCENT
            )
            
            model.to(device)
            model.eval()
            
            # Set metadata to indicate this is a rebuilt model
            if metadata:
                metadata['model_rebuilt'] = True
                metadata['rebuild_date'] = datetime.now().isoformat()
        
        # Load scaler parameters
        scaler_params = {}
        try:
            with open(scaler_path, 'r') as f:
                scaler_json = json.load(f)
                
                # Convert lists to numpy arrays
                for key in scaler_json:
                    if isinstance(scaler_json[key], list):
                        scaler_params[key] = np.array(scaler_json[key])
                    else:
                        scaler_params[key] = scaler_json[key]
        except Exception as e:
            logger.error(f"Error loading scaler parameters for {symbol}: {str(e)}")
            return model, None, metadata
        
        return model, scaler_params, metadata
        
    except Exception as e:
        logger.error(f"Unexpected error loading model for {symbol}: {str(e)}")
        return None, None, None
