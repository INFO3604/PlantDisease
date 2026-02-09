"""CNN model training runner."""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.plantdisease import config
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Trainer class for CNN models."""
    
    def __init__(self, model, device=None, checkpoint_dir=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Where to save checkpoints
        """
        self.model = model
        self.device = device or config.DEVICE
        self.checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader, loss_fn, optimizer):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training
            loss_fn: Loss function
            optimizer: Optimizer
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader, loss_fn):
        """
        Validate model.
        
        Args:
            val_loader: DataLoader for validation
            loss_fn: Loss function
        
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = loss_fn(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, optimizer, loss, checkpoint_name="checkpoint.pt"):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            optimizer: Optimizer state
            loss: Current loss
            checkpoint_name: Filename for checkpoint
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
        
        Returns:
            Epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']

def train_model(model=None, train_loader=None, val_loader=None, 
                epochs=None, learning_rate=None):
    """
    Full training pipeline.
    
    Args:
        model: PyTorch model (create default if None)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of epochs (default: config.EPOCHS)
        learning_rate: Learning rate (default: config.LEARNING_RATE)
    """
    if epochs is None:
        epochs = config.EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    logger.info(f"Starting training: {epochs} epochs, LR={learning_rate}")
    
    # TODO: Implement training loop
    #   1. Create model if not provided
    #   2. Set up optimizer and loss function
    #   3. Training loop with validation
    #   4. Save checkpoints and best model
