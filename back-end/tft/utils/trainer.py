import torch
import torch.optim as optim
import torch.nn as nn
import time
import os
import json
from config import Config

class Trainer:
    def __init__(self, model, dataloader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.progress_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'progress.json')
        
    def update_progress(self, progress):
        """Write progress to a file for the API to read"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({"progress": progress}, f)
                f.flush()
            print(f"Updated progress: {progress}%")
        except Exception as e:
            print(f"Failed to update progress: {e}")
            
    def train(self, epochs):
        start_time = time.time()
        self.model.train()
        total_steps = epochs * len(self.dataloader)
        current_step = 0
        
        # Initialize progress file
        self.update_progress(0)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                current_step += 1
                
                # Calculate and update progress more frequently
                progress = int((current_step / total_steps) * 100)
                if batch_idx % 2 == 0 or batch_idx == len(self.dataloader) - 1:
                    self.update_progress(progress)
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.dataloader)} | "
                          f"Progress: {progress}% | Loss: {loss.item():.6f}")
                
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            elapsed = time.time() - start_time
            
            # Update progress at the end of each epoch
            progress = int(((epoch + 1) / epochs) * 100)
            self.update_progress(progress)
            
            print(f"Epoch {epoch+1}/{epochs} completed | Avg Loss: {avg_epoch_loss:.6f} | "
                  f"Time: {elapsed:.2f}s | Progress: {progress}%")
        
        # Ensure progress is set to 100% when training completes
        self.update_progress(100)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
