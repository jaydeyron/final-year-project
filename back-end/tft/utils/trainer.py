import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
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
        
        # Enhanced loss function - combine MSE with direction penalty
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()  # Add MAE loss for better robustness
        
        # Use RAdam optimizer for better convergence
        try:
            # If torch_optimizer is available
            from torch_optimizer import RAdam
            self.optimizer = RAdam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=1e-4
            )
        except ImportError:
            # Fall back to Adam if torch_optimizer is not available
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
        
        # Cosine annealing scheduler for better optimization
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Multiply period by 2 after each restart
            eta_min=learning_rate/20  # Min learning rate
        )
        
        # Set up progress file path
        self.progress_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'progress.json'
        )
        
    def train(self, num_epochs):
        # Remove early stopping variables and keep only what's needed
        # best_loss = float('inf')
        # early_stopping_count = 0
        # early_stopping_patience = 15  # No longer needed
        
        # For reproducing results
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Initialize progress tracking
        self._update_progress(0)
        
        # For adaptive weighting
        direction_weight = 0.2
        huber_weight = 0.3
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Multiple loss components for better training
                # 1. MSE Loss - basic prediction accuracy
                mse_loss = self.mse_criterion(outputs.squeeze(), targets)
                
                # 2. MAE Loss - less sensitive to outliers
                mae_loss = self.mae_criterion(outputs.squeeze(), targets)
                
                # 3. Huber Loss - combination of MSE and MAE properties
                huber_loss = F.smooth_l1_loss(outputs.squeeze(), targets)
                
                # 4. Direction prediction penalty - penalize predicting wrong trend direction
                last_price = inputs[:, -1, 3]  # Last known close price
                pred_direction = ((outputs.squeeze() - last_price) > 0).float()
                target_direction = ((targets - last_price) > 0).float()
                direction_match = (pred_direction == target_direction).float()
                direction_penalty = (1 - direction_match.mean())
                
                # 5. Relative error penalty - penalize larger % errors more
                relative_error = torch.abs((outputs.squeeze() - targets) / (targets + 1e-8))
                relative_penalty = torch.mean(torch.min(relative_error, torch.ones_like(relative_error)*0.5))
                
                # Combine losses with adaptive weights
                loss = (
                    0.4 * mse_loss + 
                    0.2 * mae_loss + 
                    huber_weight * huber_loss + 
                    direction_weight * direction_penalty +
                    0.1 * relative_penalty
                )
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Add batch loss (use MSE for reporting)
                total_loss += mse_loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(self.dataloader)
            
            # Update direction weight based on accuracy (give it more weight if direction accuracy is low)
            if direction_match.mean().item() < 0.7:
                direction_weight = min(0.5, direction_weight + 0.02)
            else:
                direction_weight = max(0.1, direction_weight - 0.01)
                
            # Gradually increase huber weight for better final convergence
            huber_weight = min(0.5, huber_weight + 0.01)
            
            # Validate if we have a validation set
            val_loss = None
            if self.val_dataloader:
                val_loss = self._validate()
                # Update learning rate based on scheduler
                self.scheduler.step()
                
                # Early stopping check - remove this entire block
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     early_stopping_count = 0
                # else:
                #     early_stopping_count += 1
                #     
                # if early_stopping_count >= early_stopping_patience:
                #     print(f"Early stopping at epoch {epoch+1}")
                #     # Update progress to 100% when we early stop
                #     self._update_progress(100)
                #     break
            
            # Log progress
            progress = int((epoch + 1) / num_epochs * 100)
            self._update_progress(progress)
            
            # Adjusted print statement - only show val loss when available
            if val_loss is not None:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, Dir Weight: {direction_weight:.2f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Dir Weight: {direction_weight:.2f}')
        
        # Ensure 100% progress on completion
        self._update_progress(100)
    
    def _validate(self):
        """Run validation with enhanced metrics"""
        self.model.eval()
        total_mse_loss = 0
        direction_accuracy = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                last_price = inputs[:, -1, 3]  # Last known close price
                outputs = self.model(inputs)
                
                # Calculate MSE loss
                loss = self.mse_criterion(outputs.squeeze(), targets)
                total_mse_loss += loss.item() * inputs.size(0)
                
                # Calculate direction accuracy
                pred_direction = (outputs.squeeze() > last_price).float()
                target_direction = (targets > last_price).float()
                direction_match = (pred_direction == target_direction).float().sum().item()
                direction_accuracy += direction_match
                total_samples += inputs.size(0)
        
        # Log more detailed validation metrics
        avg_mse = total_mse_loss / total_samples
        avg_direction_acc = direction_accuracy / total_samples
        print(f"Validation - MSE: {avg_mse:.6f}, Direction Accuracy: {avg_direction_acc:.2f}")
                
        return avg_mse  # Return MSE as the primary metric for early stopping

    def _update_progress(self, progress, error=None):
        """Update the progress file with the current training progress with better error handling"""
        try:
            # Create the progress data with safer error handling
            progress_data = {"progress": progress}
            if error is not None:
                # Safely convert error to string
                error_message = str(error) if error is not None else "Unknown error"
                # Ensure we don't use format specifiers in error message
                error_message = error_message.replace('{', '{{').replace('}', '}}')
                progress_data["error"] = error_message
            
            # Use a temporary file to ensure atomic write
            progress_dir = os.path.dirname(self.progress_file)
            temp_file = os.path.join(progress_dir, f'progress_temp_{os.getpid()}.json')
                
            # Write to temp file first
            with open(temp_file, 'w') as f:
                json.dump(progress_data, f)
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
            print(f"Error updating progress: {str(e) if e is not None else 'Unknown error'}")
            
            # As a last resort, try a direct write
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f)
            except Exception as direct_error:
                print(f"Direct progress update also failed: {str(direct_error) if direct_error is not None else 'Unknown error'}")
