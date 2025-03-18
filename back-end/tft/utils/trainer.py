import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, train_loader, device, learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def train(self, epochs):
        print(f"\nStarting training for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_data, batch_targets in self.train_loader:
                batch_data, batch_targets = batch_data.to(self.device), batch_targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.scheduler.step(avg_loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")