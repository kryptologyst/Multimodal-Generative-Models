"""Unit tests for multimodal generative models."""

import pytest
import torch
from PIL import Image
import numpy as np

from src.utils.device import get_device, set_seed, get_device_info
from src.utils.config import Config
from src.data.dataset import MultimodalDataset, create_sample_dataset
from src.eval.metrics import MultimodalEvaluator


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are deterministic
        rand1 = torch.rand(1)
        set_seed(42)
        rand2 = torch.rand(1)
        assert torch.allclose(rand1, rand2)
    
    def test_get_device_info(self):
        """Test device info retrieval."""
        info = get_device_info()
        assert "device" in info
        assert "device_type" in info
        assert info["device_type"] in ["cuda", "mps", "cpu"]


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = Config()
        assert config.get("seed") == 42
        assert config.get("device") == "auto"
        assert "model" in config.to_dict()
    
    def test_config_get_set(self):
        """Test configuration get/set operations."""
        config = Config()
        config.set("test_value", 123)
        assert config.get("test_value") == 123
        assert config.get("nonexistent", "default") == "default"
    
    def test_config_update(self):
        """Test configuration updates."""
        config = Config()
        config.update({"test": {"nested": "value"}})
        assert config.get("test.nested") == "value"


class TestDataset:
    """Test dataset functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = MultimodalDataset("nonexistent_path")
        assert len(dataset) > 0
        
        # Test getting a sample
        sample = dataset[0]
        assert "image" in sample
        assert "text" in sample
        assert "image_id" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["text"], str)
    
    def test_dataset_transform(self):
        """Test dataset transforms."""
        dataset = MultimodalDataset("nonexistent_path")
        sample = dataset[0]
        
        # Check image tensor shape
        assert sample["image"].shape[0] == 3  # RGB channels
        assert sample["image"].shape[1] == dataset.image_size
        assert sample["image"].shape[2] == dataset.image_size


class TestEvaluator:
    """Test evaluation metrics."""
    
    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = MultimodalEvaluator()
        assert evaluator.device is not None
        assert evaluator.clip_model is not None
    
    def test_clip_score(self):
        """Test CLIP score computation."""
        evaluator = MultimodalEvaluator()
        
        # Create test images and texts
        images = [Image.new('RGB', (224, 224), color='red') for _ in range(2)]
        texts = ["a red image", "a blue image"]
        
        results = evaluator.compute_clip_score(images, texts)
        
        assert "clip_score_mean" in results
        assert "clip_score_std" in results
        assert isinstance(results["clip_score_mean"], float)
        assert 0 <= results["clip_score_mean"] <= 1
    
    def test_aesthetic_score(self):
        """Test aesthetic score computation."""
        evaluator = MultimodalEvaluator()
        
        # Create test images
        images = [Image.new('RGB', (224, 224), color='red') for _ in range(2)]
        
        results = evaluator.compute_aesthetic_score(images)
        
        assert "aesthetic_score_mean" in results
        assert "aesthetic_score_std" in results
        assert isinstance(results["aesthetic_score_mean"], float)
        assert 0 <= results["aesthetic_score_mean"] <= 1
    
    def test_diversity_score(self):
        """Test diversity score computation."""
        evaluator = MultimodalEvaluator()
        
        # Create test images
        images = [Image.new('RGB', (224, 224), color='red') for _ in range(3)]
        
        score = evaluator.compute_diversity_score(images)
        
        assert isinstance(score, float)
        assert score >= 0


class TestIntegration:
    """Integration tests."""
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_sample_dataset(temp_dir)
            
            # Check that files were created
            import os
            assert os.path.exists(os.path.join(temp_dir, "train.json"))
            assert os.path.exists(os.path.join(temp_dir, "val.json"))
            assert os.path.exists(os.path.join(temp_dir, "test.json"))
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            
            # Create and save config
            config1 = Config()
            config1.set("test_value", 456)
            config1.save(config_path)
            
            # Load config
            config2 = Config(config_path)
            assert config2.get("test_value") == 456


if __name__ == "__main__":
    pytest.main([__file__])
