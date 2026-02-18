"""
Unit tests for CNN baseline model.

Tests:
- Model initialization with different backbones
- Forward pass and predictions
- Uncertainty threshold
- Top-k predictions
- ONNX export
- TorchScript export
"""

import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.plantdisease.models.cnn_baseline import (
    PlantDiseaseCNN,
    PlantDiseaseDataset,
    CNNTrainer
)


class TestPlantDiseaseCNN:
    """Test PlantDiseaseCNN model."""
    
    @pytest.mark.parametrize('backbone', [
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'efficientnet_b0'
    ])
    def test_model_initialization(self, backbone):
        """Test model initialization with different backbones."""
        model = PlantDiseaseCNN(
            num_classes=10,
            backbone=backbone,
            pretrained=False
        )
        assert model.num_classes == 10
        assert model.backbone_name == backbone
        assert model.uncertainty_threshold == 0.5
    
    @pytest.mark.parametrize('backbone', [
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'efficientnet_b0'
    ])
    def test_forward_pass(self, backbone):
        """Test forward pass through model."""
        model = PlantDiseaseCNN(
            num_classes=5,
            backbone=backbone,
            pretrained=False
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()
    
    def test_uncertainty_threshold(self):
        """Test uncertainty threshold functionality."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False,
            uncertainty_threshold=0.5
        )
        model.eval()
        
        # Create logits with known probabilities
        # [0.6, 0.3, 0.1] -> max = 0.6 (certain)
        logits_certain = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
        
        # [0.4, 0.3, 0.3] -> max = 0.4 (uncertain)
        logits_uncertain = torch.tensor([[0.5, 0.4, 0.4]], dtype=torch.float32)
        
        with torch.no_grad():
            pred_certain = model.predict(logits_certain, return_uncertainty=True)
            pred_uncertain = model.predict(logits_uncertain, return_uncertainty=True)
        
        assert pred_certain['is_uncertain'] == False
        assert pred_uncertain['is_uncertain'] == True
    
    def test_top_k_predictions(self):
        """Test top-k predictions."""
        model = PlantDiseaseCNN(
            num_classes=5,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        model.eval()
        
        # Create logits
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        
        with torch.no_grad():
            pred = model.predict(logits, k=3)
        
        assert pred['top_k_classes'].shape == (3,)
        assert pred['top_k_probs'].shape == (3,)
        assert np.allclose(pred['top_k_probs'].sum(), 1.0, atol=1e-6)
        assert np.all(np.diff(pred['top_k_probs']) <= 1e-5)  # Descending order
    
    def test_onnx_export(self):
        """Test ONNX export."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / 'model.onnx'
            model.export_onnx(export_path)
            
            assert export_path.exists()
            assert export_path.stat().st_size > 0
    
    def test_torchscript_export(self):
        """Test TorchScript export."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / 'model.pt'
            model.export_torchscript(export_path, method='trace')
            
            assert export_path.exists()
            assert export_path.stat().st_size > 0
            
            # Test loading and inference
            loaded_model = torch.jit.load(export_path)
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = loaded_model(x)
            assert output.shape == (1, 3)


class TestPlantDiseaseDataset:
    """Test PlantDiseaseDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create dummy dataset structure
            for class_name in ['class1', 'class2']:
                class_dir = tmpdir / class_name
                class_dir.mkdir()
                
                # Create dummy images
                for i in range(3):
                    img = Image.new('RGB', (224, 224), color=(i, i, i))
                    img.save(class_dir / f'image_{i}.jpg')
            
            dataset = PlantDiseaseDataset(tmpdir)
            
            assert len(dataset) == 6
            assert len(dataset.classes) == 2
            assert 'class1' in dataset.classes
            assert 'class2' in dataset.classes
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create dummy dataset structure
            class_dir = tmpdir / 'class1'
            class_dir.mkdir()
            
            img = Image.new('RGB', (224, 224), color=(100, 100, 100))
            img.save(class_dir / 'image_0.jpg')
            
            dataset = PlantDiseaseDataset(tmpdir)
            
            image, label = dataset[0]
            
            assert isinstance(image, type(img))
            assert isinstance(label, int)
            assert label == 0


class TestCNNTrainer:
    """Test CNNTrainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CNNTrainer(
                model=model,
                device='cpu',
                checkpoint_dir=tmpdir,
                export_dir=tmpdir
            )
            
            assert trainer.model == model
            assert trainer.device == 'cpu'
            assert Path(tmpdir).exists()
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CNNTrainer(
                model=model,
                device='cpu',
                checkpoint_dir=tmpdir
            )
            
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
            trainer.save_checkpoint(
                epoch=1,
                optimizer=optimizer,
                loss=0.5,
                checkpoint_path=checkpoint_path
            )
            
            assert checkpoint_path.exists()
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert checkpoint['epoch'] == 1
            assert checkpoint['loss'] == 0.5
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Modify model state
        original_weight = model.classifier[4].weight.clone()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
            
            # Save checkpoint
            torch.save({
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.5,
                'model_config': {
                    'num_classes': 3,
                    'backbone': 'mobilenet_v3_small',
                    'uncertainty_threshold': 0.5
                }
            }, checkpoint_path)
            
            # Create new model and trainer
            new_model = PlantDiseaseCNN(
                num_classes=3,
                backbone='mobilenet_v3_small',
                pretrained=False
            )
            trainer = CNNTrainer(model=new_model, device='cpu', checkpoint_dir=tmpdir)
            
            # Load checkpoint
            trainer.load_checkpoint(checkpoint_path)
            
            # Verify weights are loaded
            assert torch.allclose(
                trainer.model.classifier[4].weight,
                original_weight
            )
    
    def test_export_model(self):
        """Test model export."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CNNTrainer(
                model=model,
                device='cpu',
                checkpoint_dir=tmpdir,
                export_dir=tmpdir
            )
            
            export_paths = trainer.export_model(
                export_formats=['onnx', 'torchscript'],
                model_name='test_model'
            )
            
            assert 'onnx' in export_paths
            assert 'torchscript' in export_paths
            assert Path(export_paths['onnx']).exists()
            assert Path(export_paths['torchscript']).exists()
    
    def test_save_training_config(self):
        """Test training config saving."""
        model = PlantDiseaseCNN(
            num_classes=3,
            backbone='mobilenet_v3_small',
            pretrained=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CNNTrainer(
                model=model,
                device='cpu',
                checkpoint_dir=tmpdir
            )
            
            config_path = Path(tmpdir) / 'config.json'
            class_names = ['class1', 'class2', 'class3']
            
            trainer.save_training_config(
                config_path=config_path,
                class_names=class_names,
                config_dict={'epochs': 50, 'batch_size': 32}
            )
            
            assert config_path.exists()
            
            # Verify config contents
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            assert config['model']['num_classes'] == 3
            assert config['model']['class_names'] == class_names
            assert config['training']['epochs'] == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
