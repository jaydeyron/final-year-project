
#!/usr/bin/env python3
"""
Script to fix scalers that have incorrect number of features
"""
import os
import json
import sys
from config import Config

def fix_scaler_file(symbol):
    """Fix the scaler file for the given symbol by ensuring it has 10 features"""
    # Get model paths
    paths = Config.get_model_paths(symbol)
    scaler_path = paths['scaler']
    
    if not os.path.exists(scaler_path):
        print(f"No scaler file found for {symbol}")
        return False
    
    try:
        # Read existing scaler
        with open(scaler_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"Empty scaler file for {symbol}")
                return False
                
            scaler_data = json.loads(content)
            
        # Process mean values
        mean_vals = []
        if 'mean_' in scaler_data:
            if isinstance(scaler_data['mean_'], dict):
                # Convert dict to list
                mean_vals = [float(scaler_data['mean_'].get(str(i), 0.0)) 
                         for i in range(len(scaler_data['mean_']))]
            elif isinstance(scaler_data['mean_'], list):
                mean_vals = [float(x) for x in scaler_data['mean_']]
            else:
                # Single value, make it a list
                mean_vals = [float(scaler_data['mean_'])]
                
        # Process scale values
        scale_vals = []
        if 'scale_' in scaler_data:
            if isinstance(scaler_data['scale_'], dict):
                scale_vals = [float(scaler_data['scale_'].get(str(i), 1.0)) 
                          for i in range(len(scaler_data['scale_']))]
            elif isinstance(scaler_data['scale_'], list):
                scale_vals = [float(x) for x in scaler_data['scale_']]
            else:
                # Single value, make it a list
                scale_vals = [float(scaler_data['scale_'])]
        
        # Check if we need to extend arrays
        needs_fixing = len(mean_vals) < 10 or len(scale_vals) < 10
        
        if needs_fixing:
            # Backup original file
            backup_path = f"{scaler_path}.bak"
            os.rename(scaler_path, backup_path)
            print(f"Backed up original scaler to {backup_path}")
            
            # Extend arrays to 10 features
            if len(mean_vals) < 10:
                print(f"Extending mean from {len(mean_vals)} to 10 features")
                mean_vals.extend([0.0] * (10 - len(mean_vals)))
                
            if len(scale_vals) < 10:
                print(f"Extending scale from {len(scale_vals)} to 10 features")
                scale_vals.extend([1.0] * (10 - len(scale_vals)))
            
            # Ensure no zero values in scale
            scale_vals = [max(0.0001, x) for x in scale_vals]
            
            # Write updated scaler
            updated_scaler = {
                'mean_': mean_vals,
                'scale_': scale_vals
            }
            
            with open(scaler_path, 'w') as f:
                json.dump(updated_scaler, f, indent=2)
                
            print(f"Fixed scaler file for {symbol}")
            return True
        else:
            print(f"Scaler for {symbol} already has enough features")
            return False
            
    except Exception as e:
        print(f"Error fixing scaler for {symbol}: {str(e)}")
        return False

def fix_all_scalers():
    """Fix all scaler files in the models directory"""
    # Get the models directory
    models_dir = Config.MODEL_DIR
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
        
    # Get all subdirectories (one per symbol)
    fixed_count = 0
    error_count = 0
    
    for symbol_dir in os.listdir(models_dir):
        symbol_path = os.path.join(models_dir, symbol_dir)
        if os.path.isdir(symbol_path):
            try:
                print(f"\nChecking {symbol_dir}...")
                if fix_scaler_file(symbol_dir):
                    fixed_count += 1
            except Exception as e:
                print(f"Error processing {symbol_dir}: {str(e)}")
                error_count += 1
    
    print(f"\nFixed {fixed_count} scaler files with {error_count} errors")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Fix specific symbol
        fix_scaler_file(sys.argv[1])
    else:
        # Fix all scalers
        fix_all_scalers()