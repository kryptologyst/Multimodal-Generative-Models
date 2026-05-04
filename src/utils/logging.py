"""Logging utilities for the multimodal generative models project."""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files.
        log_level: Logging level.
        log_to_file: Whether to log to file.
        log_to_console: Whether to log to console.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create logger
    logger = logging.getLogger("multimodal_generative")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "multimodal_generative.log")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: Path):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value.
        
        Args:
            tag: Tag for the scalar.
            value: Scalar value.
            step: Step number.
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """Log image.
        
        Args:
            tag: Tag for the image.
            image: Image tensor.
            step: Step number.
        """
        self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """Log multiple images.
        
        Args:
            tag: Tag for the images.
            images: Images tensor.
            step: Step number.
        """
        self.writer.add_images(tag, images, step)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text.
        
        Args:
            tag: Tag for the text.
            text: Text to log.
            step: Step number.
        """
        self.writer.add_text(tag, text, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log histogram.
        
        Args:
            tag: Tag for the histogram.
            values: Values to histogram.
            step: Step number.
        """
        self.writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name. If None, returns root logger.
        
    Returns:
        logging.Logger: Logger instance.
    """
    if name:
        return logging.getLogger(f"multimodal_generative.{name}")
    return logging.getLogger("multimodal_generative")
