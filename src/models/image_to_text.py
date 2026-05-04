"""Image-to-text generation models using CLIP and BLIP."""

import warnings
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ImageToTextGenerator:
    """Image-to-text generation using CLIP and BLIP models."""
    
    def __init__(
        self,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        blip_model_id: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
    ):
        """Initialize the image-to-text generator.
        
        Args:
            clip_model_id: Hugging Face model ID for CLIP.
            blip_model_id: Hugging Face model ID for BLIP.
            device: Device to run on. If None, auto-detects.
        """
        self.device = device or get_device()
        
        logger.info(f"Loading CLIP model: {clip_model_id}")
        logger.info(f"Loading BLIP model: {blip_model_id}")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_id)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        
        # Load BLIP model and processor
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id)
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_id)
        
        # Move models to device
        self.clip_model = self.clip_model.to(self.device)
        self.blip_model = self.blip_model.to(self.device)
        
        # Set models to eval mode
        self.clip_model.eval()
        self.blip_model.eval()
        
        logger.info("Image-to-text generator initialized successfully")
    
    def generate_caption(
        self,
        image: Union[Image.Image, List[Image.Image]],
        max_length: int = 50,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> Union[str, List[str]]:
        """Generate captions for images using BLIP.
        
        Args:
            image: PIL Image(s) to caption.
            max_length: Maximum length of generated caption.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            do_sample: Whether to use sampling.
            
        Returns:
            Generated caption(s).
        """
        if isinstance(image, Image.Image):
            images = [image]
            return_single = True
        else:
            images = image
            return_single = False
        
        captions = []
        
        for img in images:
            try:
                # Process image
                inputs = self.blip_processor(
                    img, return_tensors="pt"
                ).to(self.device)
                
                # Generate caption
                with torch.no_grad():
                    generated_ids = self.blip_model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.blip_processor.tokenizer.pad_token_id,
                    )
                
                # Decode caption
                caption = self.blip_processor.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                captions.append(caption)
                
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                captions.append("Error generating caption")
        
        return captions[0] if return_single else captions
    
    def compute_similarity(
        self,
        image: Union[Image.Image, List[Image.Image]],
        text: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute CLIP similarity between images and text.
        
        Args:
            image: PIL Image(s).
            text: Text prompt(s).
            
        Returns:
            Similarity score(s) between 0 and 1.
        """
        if isinstance(image, Image.Image):
            images = [image]
            return_single = True
        else:
            images = image
            return_single = False
        
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        similarities = []
        
        for img in images:
            img_similarities = []
            
            for txt in texts:
                try:
                    # Process inputs
                    inputs = self.clip_processor(
                        text=txt, images=img, return_tensors="pt", padding=True
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
                        
                        img_similarities.append(similarity)
                
                except Exception as e:
                    logger.error(f"Error computing similarity: {str(e)}")
                    img_similarities.append(0.0)
            
            similarities.append(img_similarities[0] if return_single else img_similarities)
        
        return similarities[0] if return_single else similarities
    
    def find_best_match(
        self,
        image: Image.Image,
        text_candidates: List[str],
    ) -> tuple[str, float]:
        """Find the best matching text for an image.
        
        Args:
            image: PIL Image.
            text_candidates: List of text candidates.
            
        Returns:
            Tuple of (best_text, similarity_score).
        """
        similarities = self.compute_similarity(image, text_candidates)
        
        if isinstance(similarities, float):
            similarities = [similarities]
        
        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        return text_candidates[best_idx], similarities[best_idx]
    
    def generate_descriptive_caption(
        self,
        image: Image.Image,
        style: str = "detailed",
    ) -> str:
        """Generate a descriptive caption with different styles.
        
        Args:
            image: PIL Image to caption.
            style: Style of caption ("detailed", "simple", "artistic").
            
        Returns:
            Generated descriptive caption.
        """
        style_prompts = {
            "detailed": "Describe this image in detail, including colors, objects, composition, and atmosphere.",
            "simple": "Describe this image briefly and simply.",
            "artistic": "Describe this image from an artistic perspective, focusing on style, mood, and composition.",
        }
        
        if style not in style_prompts:
            style = "detailed"
        
        # Generate base caption
        base_caption = self.generate_caption(image)
        
        # Use CLIP to find best matching style
        style_candidates = [
            f"{base_caption}",
            f"A {style} view: {base_caption}",
            f"This image shows {base_caption}",
            f"In this {style} image, we see {base_caption}",
        ]
        
        best_caption, similarity = self.find_best_match(image, style_candidates)
        
        return best_caption
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'clip_model'):
            del self.clip_model
        if hasattr(self, 'blip_model'):
            del self.blip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
