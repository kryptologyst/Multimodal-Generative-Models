"""Text-to-image generation models using Stable Diffusion."""

import warnings
from typing import List, Optional, Union

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from PIL import Image

from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TextToImageGenerator:
    """Text-to-image generation using Stable Diffusion."""
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        scheduler: str = "DPMSolverMultistepScheduler",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Initialize the text-to-image generator.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion.
            scheduler: Scheduler type for diffusion process.
            device: Device to run on. If None, auto-detects.
            torch_dtype: Data type for model weights.
        """
        self.model_id = model_id
        self.device = device or get_device()
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading Stable Diffusion model: {model_id}")
        logger.info(f"Using device: {self.device}")
        
        # Load the pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable for research purposes
            requires_safety_checker=False,
        )
        
        # Set scheduler
        self._set_scheduler(scheduler)
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_memory_efficient_attention"):
            self.pipeline.enable_memory_efficient_attention()
        
        logger.info("Text-to-image generator initialized successfully")
    
    def _set_scheduler(self, scheduler_name: str) -> None:
        """Set the diffusion scheduler.
        
        Args:
            scheduler_name: Name of the scheduler to use.
        """
        scheduler_map = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler,
        }
        
        if scheduler_name in scheduler_map:
            scheduler_class = scheduler_map[scheduler_name]
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"Set scheduler to: {scheduler_name}")
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using default")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s) for image generation.
            negative_prompt: Negative prompt(s) to avoid certain content.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Guidance scale for classifier-free guidance.
            height: Height of generated images.
            width: Width of generated images.
            num_images_per_prompt: Number of images to generate per prompt.
            seed: Random seed for reproducibility.
            
        Returns:
            List of generated PIL Images.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        logger.info(f"Generating images for prompt: {prompt}")
        
        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, ugly, "
                "bad anatomy, bad proportions, extra limbs, "
                "duplicate, morbid, mutilated, extra fingers, "
                "mutated hands, poorly drawn hands, poorly drawn face, "
                "mutation, deformed, ugly, blurry, bad proportions, "
                "extra limbs, cloned face, disfigured, out of frame, "
                "ugly, extra limbs, bad anatomy, gross proportions, "
                "malformed limbs, missing arms, missing legs, "
                "extra arms, extra legs, mutated hands, "
                "fused fingers, too many fingers, long neck"
            )
        
        try:
            with torch.autocast(self.device):
                images = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                ).images
            
            logger.info(f"Successfully generated {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            raise
    
    def generate_with_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate multiple variations of the same prompt.
        
        Args:
            prompt: Text prompt for image generation.
            num_variations: Number of variations to generate.
            seed: Base seed for generation.
            **kwargs: Additional arguments for generation.
            
        Returns:
            List of generated PIL Images.
        """
        images = []
        base_seed = seed or 42
        
        for i in range(num_variations):
            current_seed = base_seed + i
            variation_images = self.generate(
                prompt=prompt,
                seed=current_seed,
                **kwargs
            )
            images.extend(variation_images)
        
        return images
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
