import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from typing import Dict, Tuple, Optional
import onnx

class PlantDiseaseCNN:
    """CNN model for plant disease classification with mobile export support."""

    def __init__(self, model_name: str = 'mobilenetv3_small', num_classes: int = 10,
                 pretrained: bool = True, uncertainty_threshold: float = 0.5):
        self.model_name = model_name
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold  # predictions below this are flagged as uncertain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(pretrained)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def _create_model(self, pretrained: bool) -> nn.Module:
        """Load a pretrained backbone and replace the final layer to match num_classes."""
        if self.model_name == 'mobilenetv3_small':
            model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT' if pretrained else None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        elif self.model_name == 'mobilenetv3_large':
            model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT' if pretrained else None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT' if pretrained else None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return model

    def predict(self, image_tensor: torch.Tensor, top_k: int = 5) -> Dict:
        """Run inference and return top-k predictions with confidence scores."""
        self.model.eval()
        with torch.no_grad():
            # Convert raw logits to probabilities and get top-k results
            probs = torch.nn.functional.softmax(self.model(image_tensor.to(self.device)), dim=1)
            top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
            top_probs = top_probs.cpu().numpy()[0].tolist()
            top_indices = top_indices.cpu().numpy()[0].tolist()
        return {
            'predictions': [{'class_id': idx, 'confidence': prob} for idx, prob in zip(top_indices, top_probs)],
            'max_confidence': top_probs[0],
            'uncertain': top_probs[0] < self.uncertainty_threshold  # flag low-confidence predictions
        }

    def export_onnx(self, save_path: str, sample_input: torch.Tensor):
        """Export model to ONNX format for cross-platform mobile deployment."""
        self.model.eval()
        torch.onnx.export(
            self.model, sample_input.to(self.device), save_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # allow variable batch size
        )
        onnx.checker.check_model(onnx.load(save_path))  # verify the exported model is valid
        print(f"ONNX model saved and verified: {save_path}")

    def export_torchscript(self, save_path: str, sample_input: torch.Tensor):
        """Export model to TorchScript (traced) for PyTorch Mobile deployment."""
        self.model.eval()
        # Use trace (not script) — MobileNetV3 has control flow unsupported by torch.jit.script
        traced = torch.jit.trace(self.model, sample_input.to(self.device))
        traced.save(save_path)
        print(f"TorchScript model saved: {save_path}")

    def save_model(self, save_path: str):
        """Save model weights and config as a checkpoint."""
        torch.save({
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'uncertainty_threshold': self.uncertainty_threshold,
            'model_state_dict': self.model.state_dict(),
        }, save_path)
        print(f"Model saved: {save_path}")

    def load_model(self, load_path: str):
        """Load a saved checkpoint and restore model state."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model_name = checkpoint['model_name']
        self.num_classes = checkpoint['num_classes']
        self.uncertainty_threshold = checkpoint['uncertainty_threshold']
        self.model = self._create_model(pretrained=False)  # create fresh architecture
        self.model.load_state_dict(checkpoint['model_state_dict'])  # restore weights
        self.model.to(self.device)
        print(f"Model loaded: {load_path}")


class CNNTrainer:
    """Handles the training loop, validation, and learning rate scheduling."""

    def __init__(self, model: PlantDiseaseCNN, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)
        # Reduce LR by 10x if validation loss stops improving for 5 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5
        )
        self.train_losses, self.val_losses, self.val_accuracies = [], [], []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one full pass over the training set and return average loss."""
        self.model.model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(self.model.device), labels.to(self.model.device)
            self.optimizer.zero_grad()
            loss = self.model.criterion(self.model.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on the validation set, returns (avg_loss, accuracy%)."""
        self.model.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.model.device), labels.to(self.model.device)
                outputs = self.model.model(images)
                total_loss += self.model.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return total_loss / len(val_loader), 100 * correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, save_path: Optional[str] = None):
        """Full training loop — saves the best model based on validation accuracy."""
        best_val_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.scheduler.step(val_loss)  # adjust LR based on val loss plateau
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # Save checkpoint whenever validation accuracy improves
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_model(save_path)
                print(f"Best model saved: {val_acc:.2f}%")


def get_data_transforms(input_size: int = 224):
    """
    Returns train and validation transforms.
    Training applies augmentation; validation only resizes and normalizes.
    ImageNet mean/std used since we're fine-tuning pretrained ImageNet weights.
    """
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


if __name__ == "__main__":
    model = PlantDiseaseCNN(model_name='mobilenetv3_small', num_classes=10)
    trainer = CNNTrainer(model, learning_rate=0.001)

    # Uncomment and set paths to train:
    # train_tf, val_tf = get_data_transforms()
    # train_loader = DataLoader(ImageFolder('path/to/train', transform=train_tf), batch_size=32, shuffle=True)
    # val_loader = DataLoader(ImageFolder('path/to/val', transform=val_tf), batch_size=32)
    # trainer.train(train_loader, val_loader, epochs=50, save_path='best_model.pth')

    sample_input = torch.randn(1, 3, 224, 224)
    model.export_onnx('model.onnx', sample_input)
    model.export_torchscript('model.pt', sample_input)

    print(model.predict(sample_input, top_k=3))