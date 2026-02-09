"""Model evaluation and metrics computation."""
import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from src.plantdisease import config
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """Evaluator class for CNN models."""
    
    def __init__(self, model, device=None):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device for evaluation
        """
        self.model = model
        self.device = device or config.DEVICE
        self.predictions = None
        self.ground_truth = None
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for testing
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        self.predictions = np.array(all_predictions)
        self.ground_truth = np.array(all_labels)
        
        metrics = self.compute_metrics()
        return metrics
    
    def compute_metrics(self):
        """
        Compute evaluation metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        if self.predictions is None or self.ground_truth is None:
            raise ValueError("Must run evaluate() first")
        
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        precision = precision_score(self.ground_truth, self.predictions, average='weighted')
        recall = recall_score(self.ground_truth, self.predictions, average='weighted')
        f1 = f1_score(self.ground_truth, self.predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix array
        """
        if self.predictions is None or self.ground_truth is None:
            raise ValueError("Must run evaluate() first")
        
        cm = confusion_matrix(self.ground_truth, self.predictions)
        return cm
    
    def save_metrics(self, metrics, output_dir=None):
        """
        Save metrics to JSON/CSV.
        
        Args:
            metrics: Dictionary of metrics
            output_dir: Where to save (default: config.METRICS_DIR)
        """
        import json
        
        if output_dir is None:
            output_dir = config.METRICS_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_file}")

def evaluate_model(model=None, test_loader=None):
    """
    Full evaluation pipeline.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
    """
    logger.info("Starting model evaluation...")
    
    # TODO: Implement evaluation pipeline
    #   1. Load model if not provided
    #   2. Create evaluator
    #   3. Run evaluation
    #   4. Compute metrics and save results
    #   5. Generate plots/visualizations
