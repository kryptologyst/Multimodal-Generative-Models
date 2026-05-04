#!/usr/bin/env python3
"""Script to test the multimodal generative models installation."""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ Diffusers import failed: {e}")
        return False
    
    try:
        from src.app import MultimodalGenerativeApp
        print("✓ Main app import successful")
    except ImportError as e:
        print(f"✗ Main app import failed: {e}")
        return False
    
    try:
        from src.models.text_to_image import TextToImageGenerator
        print("✓ Text-to-image model import successful")
    except ImportError as e:
        print(f"✗ Text-to-image model import failed: {e}")
        return False
    
    try:
        from src.models.image_to_text import ImageToTextGenerator
        print("✓ Image-to-text model import successful")
    except ImportError as e:
        print(f"✗ Image-to-text model import failed: {e}")
        return False
    
    try:
        from src.eval.metrics import MultimodalEvaluator
        print("✓ Evaluator import successful")
    except ImportError as e:
        print(f"✗ Evaluator import failed: {e}")
        return False
    
    return True


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from src.utils.device import get_device, get_device_info
        
        device = get_device()
        print(f"✓ Detected device: {device}")
        
        info = get_device_info()
        print(f"✓ Device info: {info['device_type']}")
        
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import Config
        
        config = Config()
        assert config.get("seed") == 42
        assert config.get("device") == "auto"
        
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_data_creation():
    """Test sample data creation."""
    print("\nTesting data creation...")
    
    try:
        from src.data.dataset import create_sample_dataset
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_sample_dataset(temp_dir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "train.json"))
            assert os.path.exists(os.path.join(temp_dir, "val.json"))
            assert os.path.exists(os.path.join(temp_dir, "test.json"))
            
            print("✓ Sample data creation successful")
            return True
    except Exception as e:
        print(f"✗ Data creation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Multimodal Generative Models - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device_detection,
        test_config,
        test_data_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Run: python -m src.app --create-sample-data")
        print("2. Run: python -m src.app --demo")
        print("3. Run: streamlit run demo/app.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
