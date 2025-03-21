#!/bin/bash
# Script to fix the SBIN model scaler

cd "$(dirname "$0")"
echo "Fixing SBIN model scaler..."
python -c "
import json
import os

scaler_path = './models/SBIN/scaler.json'
if os.path.exists(scaler_path):
    # Backup existing file
    os.rename(scaler_path, f'{scaler_path}.bak')
    print(f'Backed up existing scaler to {scaler_path}.bak')
    
    # Create new scaler with 10 features
    scaler = {
        'mean_': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'scale_': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }
    
    # Save new scaler
    with open(scaler_path, 'w') as f:
        json.dump(scaler, f, indent=2)
    
    print('Created new scaler with 10 features')
else:
    print('SBIN scaler file not found')
"

echo "Now run the following command to verify the fix:"
echo "curl -X POST http://localhost:8000/debug-predict -H \"Content-Type: application/json\" -d '{\"symbol\":\"SBIN\"}'"
