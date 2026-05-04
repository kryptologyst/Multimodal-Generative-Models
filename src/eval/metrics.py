"""Evaluation metrics for multimodal generative models."""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchmetrics.image import FrechetInceptionDistance
from transformers import CLIPModel, CLIPProcessor

from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MultimodalEvaluator:
    """Evaluator for multimodal generative models."""
    
    def __init__(
        self,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """Initialize the evaluator.
        
        Args:
            clip_model_id: Hugging Face model ID for CLIP.
            device: Device to run on. If None, auto-detects.
        """
        self.device = device or get_device()
        
        logger.info(f"Loading CLIP model for evaluation: {clip_model_id}")
        
        # Load CLIP model for evaluation
        self.clip_model = CLIPModel.from_pretrained(clip_model_id)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        
        # Move to device
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Initialize FID metric
        self.fid_metric = FrechetInceptionDistance(feature=2048)
        
        logger.info("Multimodal evaluator initialized successfully")
    
    def compute_clip_score(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> Dict[str, float]:
        """Compute CLIP score between images and texts.
        
        Args:
            images: List of PIL Images.
            texts: List of text prompts.
            
        Returns:
            Dictionary with CLIP score metrics.
        """
        if len(images) != len(texts):
            raise ValueError("Number of images and texts must match")
        
        similarities = []
        
        for image, text in zip(images, texts):
            try:
                # Process inputs
                inputs = self.clip_processor(
                    text=text, images=image, return_tensors="pt", padding=True
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    
                    # Compute cosine similarity
                    similarity = F.cosine_similarity(
                        image_embeds, text_embeds, dim=1
                    ).item()
                    
                    similarities.append(similarity)
            
            except Exception as e:
                logger.error(f"Error computing CLIP score: {str(e)}")
                similarities.append(0.0)
        
        similarities = np.array(similarities)
        
        return {
            "clip_score_mean": float(np.mean(similarities)),
            "clip_score_std": float(np.std(similarities)),
            "clip_score_min": float(np.min(similarities)),
            "clip_score_max": float(np.max(similarities)),
        }
    
    def compute_fid(
        self,
        real_images: List[Image.Image],
        generated_images: List[Image.Image],
    ) -> float:
        """Compute Fréchet Inception Distance (FID).
        
        Args:
            real_images: List of real PIL Images.
            generated_images: List of generated PIL Images.
            
        Returns:
            FID score (lower is better).
        """
        try:
            # Convert PIL images to tensors
            real_tensors = self._pil_to_tensor(real_images)
            generated_tensors = self._pil_to_tensor(generated_images)
            
            # Update FID metric
            self.fid_metric.update(real_tensors, real=True)
            self.fid_metric.update(generated_tensors, real=False)
            
            # Compute FID
            fid_score = self.fid_metric.compute().item()
            
            # Reset metric for next computation
            self.fid_metric.reset()
            
            return fid_score
            
        except Exception as e:
            logger.error(f"Error computing FID: {str(e)}")
            return float('inf')
    
    def compute_aesthetic_score(
        self,
        images: List[Image.Image],
    ) -> Dict[str, float]:
        """Compute aesthetic score for images.
        
        This is a simplified aesthetic score based on image properties.
        For production use, consider using a dedicated aesthetic model.
        
        Args:
            images: List of PIL Images.
            
        Returns:
            Dictionary with aesthetic metrics.
        """
        scores = []
        
        for image in images:
            try:
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(image)
                
                # Compute basic aesthetic metrics
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
                
                # Color diversity (number of unique colors)
                unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
                color_diversity = unique_colors / (img_array.shape[0] * img_array.shape[1])
                
                # Edge density (simplified)
                gray = np.mean(img_array, axis=2)
                edges = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1))
                edge_density = np.mean(edges)
                
                # Combined aesthetic score (simplified)
                aesthetic_score = (
                    0.3 * (brightness / 255.0) +
                    0.3 * (contrast / 255.0) +
                    0.2 * color_diversity +
                    0.2 * (edge_density / 255.0)
                )
                
                scores.append(aesthetic_score)
                
            except Exception as e:
                logger.error(f"Error computing aesthetic score: {str(e)}")
                scores.append(0.0)
        
        scores = np.array(scores)
        
        return {
            "aesthetic_score_mean": float(np.mean(scores)),
            "aesthetic_score_std": float(np.std(scores)),
            "aesthetic_score_min": float(np.min(scores)),
            "aesthetic_score_max": float(np.max(scores)),
        }
    
    def compute_diversity_score(
        self,
        images: List[Image.Image],
    ) -> float:
        """Compute diversity score for generated images.
        
        Args:
            images: List of PIL Images.
            
        Returns:
            Diversity score (higher is better).
        """
        if len(images) < 2:
            return 0.0
        
        try:
            # Convert images to tensors
            image_tensors = self._pil_to_tensor(images)
            
            # Compute pairwise distances
            distances = []
            for i in range(len(image_tensors)):
                for j in range(i + 1, len(image_tensors)):
                    # Compute L2 distance between flattened images
                    dist = F.mse_loss(image_tensors[i], image_tensors[j]).item()
                    distances.append(dist)
            
            # Diversity is the mean pairwise distance
            diversity = np.mean(distances) if distances else 0.0
            
            return float(diversity)
            
        except Exception as e:
            logger.error(f"Error computing diversity score: {str(e)}")
            return 0.0
    
    def evaluate_generation(
        self,
        generated_images: List[Image.Image],
        prompts: List[str],
        real_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, float]:
        """Comprehensive evaluation of generated images.
        
        Args:
            generated_images: List of generated PIL Images.
            prompts: List of text prompts used for generation.
            real_images: Optional list of real images for FID computation.
            
        Returns:
            Dictionary with all evaluation metrics.
        """
        results = {}
        
        # CLIP Score
        logger.info("Computing CLIP score...")
        clip_results = self.compute_clip_score(generated_images, prompts)
        results.update(clip_results)
        
        # Aesthetic Score
        logger.info("Computing aesthetic score...")
        aesthetic_results = self.compute_aesthetic_score(generated_images)
        results.update(aesthetic_results)
        
        # Diversity Score
        logger.info("Computing diversity score...")
        diversity_score = self.compute_diversity_score(generated_images)
        results["diversity_score"] = diversity_score
        
        # FID Score (if real images provided)
        if real_images is not None:
            logger.info("Computing FID score...")
            fid_score = self.compute_fid(real_images, generated_images)
            results["fid_score"] = fid_score
        
        logger.info("Evaluation completed successfully")
        return results
    
    def _pil_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert PIL images to tensor format.
        
        Args:
            images: List of PIL Images.
            
        Returns:
            Tensor of images.
        """
        tensors = []
        for image in images:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to tensor
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'clip_model'):
            del self.clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
