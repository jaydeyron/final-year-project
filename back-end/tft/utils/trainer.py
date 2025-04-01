import torch
import torch.nn as nn
import json
import os
import time
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config

class Trainer:
    def __init__(self, model, dataloader, device, learning_rate=0.001, val_dataloader=None):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.model.to(device)
        self.criterion = nn.MSELoss()
        
        # Use Adam optimizer with weight decay to prevent overfitting
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        # Learning rate scheduler to reduce LR when validation loss plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Set up progress file path
        self.progress_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'progress.json'
        )
        
    def train(self, num_epochs):
        best_loss = float('inf')
        early_stopping_count = 0
        early_stopping_patience = 15  # Number of epochs with no improvement before stopping
        
        # For reproducing results
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Initialize progress tracking
        self._update_progress(0)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs.squeeze(), targets)
                
                # Add directional prediction penalty
                # This penalizes the model for predicting the wrong direction
                pred_direction = ((outputs.squeeze() - inputs[:, -1, 3]) > 0).float()
                target_direction = ((targets - inputs[:, -1, 3]) > 0).float()
                direction_match = (pred_direction == target_direction).float()
                direction_penalty = 0.1 * (1 - direction_match.mean())
                
                # Combine losses
                total_batch_loss = loss + direction_penalty
                
                # Backward pass
                total_batch_loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Add batch loss
                total_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(self.dataloader)
            
            # Validate if we have a validation set
            val_loss = None
            if self.val_dataloader:
                val_loss = self._validate()
                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    
                if early_stopping_count >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Update progress to 100% when we early stop
                    self._update_progress(100)
                    break
            
            # Log progress
            progress = int((epoch + 1) / num_epochs * 100)
            self._update_progress(progress)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f if val_loss else "N/A"}')
        
        # Ensure 100% progress on completion
        self._update_progress(100)
    
    def _validate(self):
        """Run validation on the validation dataset"""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                total_val_loss += loss.item()
                
        return total_val_loss / len(self.val_dataloader)
    
    def _update_progress(self, progress):
        """Update the progress file with the current training progress with better error handling"""
        try:
            # Use a temporary file to ensure atomic write
            progress_dir = os.path.dirname(self.progress_file)
            temp_file = os.path.join(progress_dir, f'progress_temp_{os.getpid()}.json')
                
            # Write to temp file first
            with open(temp_file, 'w') as f:
                json.dump({"progress": progress}, f)
                f.flush()
                os.fsync(f.fileno())
                
            # Then rename (atomic operation on most file systems)
            if os.path.exists(temp_file):
                if os.name == 'nt':  # For Windows
                    # Windows may need file removal first
                    if os.path.exists(self.progress_file):
                        os.remove(self.progress_file)
                os.rename(temp_file, self.progress_file)
                
            # Verify the file was updated properly
            if os.path.exists(self.progress_file):
                try:
                    with open(self.progress_file, 'r') as f:
                        content = json.load(f)
                        if content.get("progress") != progress:
                            print(f"Warning: Progress file contains {content.get('progress')} instead of {progress}")
                except Exception as e:
                    print(f"Warning: Could not verify progress file: {e}")
                    
        except Exception as e:
            print(f"Error updating progress: {e}")
            
            # As a last resort, try a direct write
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump({"progress": progress}, f)
            except Exception as direct_error:
                print(f"Direct progress update also failed: {direct_error}")
