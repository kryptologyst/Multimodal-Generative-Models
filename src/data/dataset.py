"""Data handling and preprocessing for multimodal generative models."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalDataset(Dataset):
    """Dataset for multimodal text-image pairs."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        image_size: int = 512,
        max_text_length: int = 77,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset.
            split: Dataset split ("train", "val", "test").
            image_size: Size to resize images to.
            max_text_length: Maximum length of text sequences.
            transform: Optional image transforms.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Load data
        self.data = self._load_data()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict]:
        """Load dataset from directory structure or JSON file.
        
        Returns:
            List of data samples.
        """
        # Try to load from JSON file first
        json_path = self.data_dir / f"{self.split}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded data from {json_path}")
            return data
        
        # Try to load from directory structure
        images_dir = self.data_dir / "images" / self.split
        texts_dir = self.data_dir / "texts" / self.split
        
        if images_dir.exists() and texts_dir.exists():
            data = self._load_from_directories(images_dir, texts_dir)
            logger.info(f"Loaded data from directories: {images_dir}, {texts_dir}")
            return data
        
        # Create synthetic data if no real data exists
        logger.warning("No data found, creating synthetic dataset")
        return self._create_synthetic_data()
    
    def _load_from_directories(
        self, images_dir: Path, texts_dir: Path
    ) -> List[Dict]:
        """Load data from image and text directories.
        
        Args:
            images_dir: Directory containing images.
            texts_dir: Directory containing text files.
            
        Returns:
            List of data samples.
        """
        data = []
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for image_file in image_files:
            # Find corresponding text file
            text_file = texts_dir / f"{image_file.stem}.txt"
            
            if text_file.exists():
                with open(text_file, 'r') as f:
                    text = f.read().strip()
                
                data.append({
                    "image_path": str(image_file),
                    "text": text,
                    "image_id": image_file.stem,
                })
        
        return data
    
    def _create_synthetic_data(self) -> List[Dict]:
        """Create synthetic data for testing.
        
        Returns:
            List of synthetic data samples.
        """
        synthetic_prompts = [
            "A beautiful sunset over the ocean",
            "A cat sitting on a windowsill",
            "A modern city skyline at night",
            "A peaceful forest with tall trees",
            "A colorful garden with flowers",
            "A mountain landscape with snow",
            "A vintage car on a country road",
            "A cozy living room with a fireplace",
            "A busy marketplace with vendors",
            "A serene lake with mountains in the background",
        ]
        
        data = []
        for i, prompt in enumerate(synthetic_prompts):
            data.append({
                "image_path": None,  # Will be generated on-the-fly
                "text": prompt,
                "image_id": f"synthetic_{i}",
                "synthetic": True,
            })
        
        return data
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms.
        
        Returns:
            Compose transform pipeline.
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def _create_synthetic_image(self, text: str) -> Image.Image:
        """Create a synthetic image for testing.
        
        Args:
            text: Text prompt.
            
        Returns:
            Synthetic PIL Image.
        """
        # Create a simple colored image based on text content
        colors = {
            "sunset": (255, 165, 0),
            "cat": (128, 128, 128),
            "city": (64, 64, 64),
            "forest": (34, 139, 34),
            "garden": (50, 205, 50),
            "mountain": (139, 69, 19),
            "car": (255, 0, 0),
            "living": (139, 69, 19),
            "market": (255, 215, 0),
            "lake": (0, 191, 255),
        }
        
        # Find matching color
        color = (128, 128, 128)  # Default gray
        for keyword, col in colors.items():
            if keyword in text.lower():
                color = col
                break
        
        # Create image
        image = Image.new('RGB', (self.image_size, self.image_size), color)
        return image
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with image and text tensors.
        """
        sample = self.data[idx]
        
        # Load or create image
        if sample.get("synthetic", False):
            image = self._create_synthetic_image(sample["text"])
        else:
            image_path = sample["image_path"]
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert('RGB')
            else:
                # Fallback to synthetic image
                image = self._create_synthetic_image(sample["text"])
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Process text
        text = sample["text"]
        
        return {
            "image": image_tensor,
            "text": text,
            "image_id": sample["image_id"],
        }


class MultimodalDataLoader:
    """Data loader for multimodal datasets."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 1,
        num_workers: int = 4,
        image_size: int = 512,
        max_text_length: int = 77,
        shuffle: bool = True,
    ):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset.
            batch_size: Batch size for data loading.
            num_workers: Number of worker processes.
            image_size: Size to resize images to.
            max_text_length: Maximum length of text sequences.
            shuffle: Whether to shuffle the data.
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.shuffle = shuffle
        
        # Create datasets for different splits
        self.datasets = {}
        self.dataloaders = {}
        
        for split in ["train", "val", "test"]:
            self._create_split(split)
    
    def _create_split(self, split: str) -> None:
        """Create dataset and dataloader for a split.
        
        Args:
            split: Dataset split name.
        """
        dataset = MultimodalDataset(
            data_dir=self.data_dir,
            split=split,
            image_size=self.image_size,
            max_text_length=self.max_text_length,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if split == "train" else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if split == "train" else False,
        )
        
        self.datasets[split] = dataset
        self.dataloaders[split] = dataloader
        
        logger.info(f"Created {split} dataloader with {len(dataset)} samples")
    
    def get_dataloader(self, split: str) -> DataLoader:
        """Get dataloader for a specific split.
        
        Args:
            split: Dataset split name.
            
        Returns:
            DataLoader for the split.
        """
        if split not in self.dataloaders:
            raise ValueError(f"Unknown split: {split}")
        
        return self.dataloaders[split]
    
    def get_dataset(self, split: str) -> MultimodalDataset:
        """Get dataset for a specific split.
        
        Args:
            split: Dataset split name.
            
        Returns:
            Dataset for the split.
        """
        if split not in self.datasets:
            raise ValueError(f"Unknown split: {split}")
        
        return self.datasets[split]


def create_sample_dataset(output_dir: Union[str, Path]) -> None:
    """Create a sample dataset for testing.
    
    Args:
        output_dir: Directory to save the sample dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "texts" / split).mkdir(parents=True, exist_ok=True)
    
    # Sample data
    sample_data = {
        "train": [
            {"text": "A beautiful sunset over the ocean", "image_id": "sunset_1"},
            {"text": "A cat sitting on a windowsill", "image_id": "cat_1"},
            {"text": "A modern city skyline at night", "image_id": "city_1"},
        ],
        "val": [
            {"text": "A peaceful forest with tall trees", "image_id": "forest_1"},
            {"text": "A colorful garden with flowers", "image_id": "garden_1"},
        ],
        "test": [
            {"text": "A mountain landscape with snow", "image_id": "mountain_1"},
            {"text": "A vintage car on a country road", "image_id": "car_1"},
        ],
    }
    
    # Save JSON files
    for split, data in sample_data.items():
        json_path = output_dir / f"{split}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    logger.info(f"Created sample dataset in {output_dir}")
