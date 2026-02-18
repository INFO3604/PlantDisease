"""
CNN baseline model for plant disease detection using MobileNetV3 and EfficientNet.

Features:
- MobileNetV3-Small and Large variants
- EfficientNet-B0 backbone option
- Uncertainty threshold for low-confidence predictions
- Top-k predictions with confidence scores
- ONNX and TorchScript export for mobile deployment
- Learning rate scheduling and model checkpointing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


class ClassifierHead(nn.Module):
    """Flexible classifier head that handles both 4D and 2D input tensors."""
    
    def __init__(self, features_dim: int, num_classes: int, dropout: float = 0.2, needs_pooling: bool = False):
        super().__init__()
        self.needs_pooling = needs_pooling
        # Always initialize avgpool for TorchScript compatibility
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if needs_pooling else nn.Identity()
        self.flatten = nn.Flatten()
        self.fc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 4D (B, C, H, W) and 2D (B, C) inputs
        if x.dim() == 4:
            x = self.avgpool(x)
        x = self.flatten(x) if x.dim() > 2 else x
        return self.fc_head(x)
    
    def __getitem__(self, index: int) -> nn.Module:
        """Support indexing for backward compatibility with Sequential-like access.
        Maps indices to actual modules: 0=avgpool/Identity, 1=flatten, 2+=fc_head modules
        """
        if index == 0:
            return self.avgpool
        elif index == 1:
            return self.flatten
        else:
            return self.fc_head[index - 2]


class PlantDiseaseDataset(Dataset):
    """Dataset class for loading plant disease images from directory structure."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing class subfolders with images
            transform: Image transformations to apply
            extensions: Valid image file extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        # Load class labels and image paths
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Collect image paths
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for image_file in class_dir.iterdir():
                if image_file.suffix.lower() in self.extensions:
                    self.images.append(image_file)
                    self.labels.append(self.class_to_idx[class_name])
        
        logger.info(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        for cls_name in self.classes:
            count = sum(1 for label in self.labels if label == self.class_to_idx[cls_name])
            logger.info(f"  {cls_name}: {count} images")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and return image and label."""
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        return self.classes


class PlantDiseaseCNN(nn.Module):
    """
    CNN model for plant disease classification.
    
    Supports MobileNetV3 (Small/Large) and EfficientNet-B0 backbones.
    Includes uncertainty estimation based on confidence threshold.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'mobilenet_v3_small',
        pretrained: bool = True,
        dropout: float = 0.2,
        uncertainty_threshold: float = 0.5
    ):
        """
        Initialize CNN model.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture ('mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0')
            pretrained: Use pretrained weights
            dropout: Dropout rate for regularization
            uncertainty_threshold: Confidence threshold below which prediction is "uncertain"
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.uncertainty_threshold = uncertainty_threshold
        
        # Load backbone
        if backbone == 'mobilenet_v3_small':
            self.backbone = mobilenet_v3_small(pretrained=pretrained)
            features_dim = 576
        elif backbone == 'mobilenet_v3_large':
            self.backbone = mobilenet_v3_large(pretrained=pretrained)
            features_dim = 960
        elif backbone == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            features_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove original classifier
        if backbone.startswith('mobilenet'):
            self.backbone.classifier = nn.Identity()
            needs_pooling = False
        elif backbone == 'efficientnet_b0':
            self.backbone.classifier = nn.Identity()
            needs_pooling = True
        
        # Custom classifier head
        self.classifier = ClassifierHead(features_dim, num_classes, dropout, needs_pooling)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict(
        self,
        logits: torch.Tensor,
        k: int = 3,
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Generate predictions with uncertainty.
        
        Args:
            logits: Model output logits
            k: Number of top predictions to return
            return_uncertainty: Whether to flag uncertain predictions
        
        Returns:
            Dictionary with:
            - 'class_idx': Predicted class index
            - 'class_prob': Confidence score [0, 1]
            - 'is_uncertain': Whether confidence < threshold
            - 'top_k_classes': Top k class indices
            - 'top_k_probs': Top k confidence scores
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probs, min(k, self.num_classes), dim=-1)
        
        # Renormalize top-k probabilities to sum to 1.0
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Get maximum probability and class
        max_prob, max_idx = torch.max(probs, dim=-1)
        
        # Check uncertainty
        is_uncertain = max_prob < self.uncertainty_threshold if return_uncertainty else False
        
        return {
            'class_idx': max_idx.item() if logits.dim() == 1 else max_idx,
            'class_prob': max_prob.item() if logits.dim() == 1 else max_prob,
            'is_uncertain': is_uncertain.item() if isinstance(is_uncertain, torch.Tensor) else is_uncertain,
            'top_k_classes': top_k_indices[0].cpu().numpy() if logits.dim() > 1 else top_k_indices.cpu().numpy(),
            'top_k_probs': top_k_probs[0].cpu().numpy() if logits.dim() > 1 else top_k_probs.cpu().numpy()
        }
    
    def export_onnx(
        self,
        output_path: Union[str, Path],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 13
    ) -> None:
        """
        Export model to ONNX format for mobile deployment.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
        """
        self.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_input = torch.randn(*input_shape, device=next(self.parameters()).device)
        
        torch.onnx.export(
            self,
            dummy_input,
            str(output_path),
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info(f"ONNX model exported to {output_path}")
    
    def export_torchscript(
        self,
        output_path: Union[str, Path],
        method: str = 'script'
    ) -> None:
        """
        Export model to TorchScript format for mobile deployment.
        
        Args:
            output_path: Path to save TorchScript model
            method: Export method ('script' or 'trace')
        """
        self.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if method == 'script':
            scripted = torch.jit.script(self)
        elif method == 'trace':
            dummy_input = torch.randn(1, 3, 224, 224, device=next(self.parameters()).device)
            scripted = torch.jit.trace(self, dummy_input)
        else:
            raise ValueError(f"Unknown export method: {method}")
        
        scripted.save(str(output_path))
        logger.info(f"TorchScript model exported to {output_path}")


class CNNTrainer:
    """
    Trainer class for CNN models with learning rate scheduling and checkpointing.
    """
    
    def __init__(
        self,
        model: PlantDiseaseCNN,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        export_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: CNN model to train
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            export_dir: Directory to save exported models
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.checkpoint_dir = Path(checkpoint_dir or 'checkpoints')
        self.export_dir = Path(export_dir or 'exports')
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        scheduler_type: str = 'cosine',
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            warmup_epochs: Number of warmup epochs
            scheduler_type: LR scheduler type ('cosine', 'step', 'plateau')
            class_weights: Class weights for loss function
        
        Returns:
            Training history dictionary
        """
        # Setup loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        best_val_loss = float('inf')
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            if epoch >= warmup_epochs:
                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, optimizer, val_loss, best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(epoch, optimizer, val_loss, checkpoint_path)
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_acc
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_acc
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            class_names: Names of classes for detailed report
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_uncertain = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                
                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=1)[0]
                uncertain = max_probs < self.model.uncertainty_threshold
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertain.extend(uncertain.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_uncertain = np.array(all_uncertain)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'uncertain_rate': all_uncertain.mean(),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        if class_names:
            metrics['classification_report'] = {
                class_names[i]: {
                    'precision': precision_score(all_labels == i, all_preds == i, zero_division=0),
                    'recall': recall_score(all_labels == i, all_preds == i, zero_division=0),
                    'f1': f1_score(all_labels == i, all_preds == i, zero_division=0)
                }
                for i in range(len(class_names))
            }
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        loss: float,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Save model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'num_classes': self.model.num_classes,
                'backbone': self.model.backbone_name,
                'uncertainty_threshold': self.model.uncertainty_threshold
            }
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def export_model(
        self,
        export_formats: List[str] = ['onnx', 'torchscript'],
        model_name: str = 'plant_disease_model'
    ) -> Dict[str, Path]:
        """
        Export model in multiple formats for mobile deployment.
        
        Args:
            export_formats: List of export formats ('onnx', 'torchscript')
            model_name: Base name for exported models
        
        Returns:
            Dictionary mapping format to export path
        """
        export_paths = {}
        
        if 'onnx' in export_formats:
            onnx_path = self.export_dir / f'{model_name}.onnx'
            self.model.export_onnx(onnx_path)
            export_paths['onnx'] = onnx_path
        
        if 'torchscript' in export_formats:
            ts_path = self.export_dir / f'{model_name}.pt'
            self.model.export_torchscript(ts_path)
            export_paths['torchscript'] = ts_path
        
        logger.info(f"Models exported to {self.export_dir}")
        return export_paths
    
    def save_training_config(
        self,
        config_path: Union[str, Path],
        class_names: List[str],
        config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save training configuration and class names."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'model': {
                'backbone': self.model.backbone_name,
                'num_classes': self.model.num_classes,
                'uncertainty_threshold': self.model.uncertainty_threshold,
                'class_names': class_names
            },
            'training': config_dict or {}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
