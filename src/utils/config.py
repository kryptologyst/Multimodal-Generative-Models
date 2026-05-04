"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the multimodal generative models project."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> DictConfig:
        """Load configuration from file or create default."""
        if self.config_path and Path(self.config_path).exists():
            return OmegaConf.load(self.config_path)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration."""
        default_config = {
            "model": {
                "text_to_image": {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "scheduler": "DPMSolverMultistepScheduler",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "height": 512,
                    "width": 512,
                    "num_images_per_prompt": 1,
                },
                "image_to_text": {
                    "model_id": "openai/clip-vit-base-patch32",
                    "max_length": 77,
                },
                "clip": {
                    "model_id": "openai/clip-vit-base-patch32",
                    "device": "auto",
                }
            },
            "data": {
                "batch_size": 1,
                "num_workers": 4,
                "image_size": 512,
                "max_text_length": 77,
            },
            "training": {
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "save_every": 5,
                "eval_every": 1,
                "gradient_accumulation_steps": 1,
                "mixed_precision": True,
            },
            "evaluation": {
                "metrics": ["clip_score", "fid", "aesthetic_score"],
                "num_samples": 100,
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "outputs",
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs",
            },
            "device": "auto",
            "seed": 42,
            "safety": {
                "nsfw_filter": True,
                "content_filter": True,
                "watermark": True,
            }
        }
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self._config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary.
        
        Args:
            config_dict: Dictionary of configuration updates.
        """
        self._config.update(config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration.
        """
        OmegaConf.save(self._config, path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the underlying DictConfig object."""
        return self._config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Config: Configuration object.
    """
    return Config(config_path)
