"""Main application for multimodal generative models."""

import argparse
import random
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from src.data.dataset import MultimodalDataLoader, create_sample_dataset
from src.eval.metrics import MultimodalEvaluator
from src.models.image_to_text import ImageToTextGenerator
from src.models.text_to_image import TextToImageGenerator
from src.utils.config import Config
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logging, get_logger


class MultimodalGenerativeApp:
    """Main application for multimodal generative models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file.
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Set up logging
        self.logger = setup_logging(
            log_dir=Path(self.config.get("paths.log_dir", "logs")),
            log_level="INFO"
        )
        
        # Set seed for reproducibility
        set_seed(self.config.get("seed", 42))
        
        # Initialize device
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Initialize evaluator
        self.evaluator = MultimodalEvaluator(device=self.device)
        
        self.logger.info("Multimodal generative app initialized successfully")
    
    def _initialize_models(self) -> None:
        """Initialize the generative models."""
        self.logger.info("Initializing models...")
        
        # Text-to-image generator
        t2i_config = self.config.get("model.text_to_image", {})
        self.text_to_image = TextToImageGenerator(
            model_id=t2i_config.get("model_id", "runwayml/stable-diffusion-v1-5"),
            scheduler=t2i_config.get("scheduler", "DPMSolverMultistepScheduler"),
            device=self.device,
        )
        
        # Image-to-text generator
        i2t_config = self.config.get("model.image_to_text", {})
        self.image_to_text = ImageToTextGenerator(
            clip_model_id=i2t_config.get("clip_model_id", "openai/clip-vit-base-patch32"),
            blip_model_id=i2t_config.get("blip_model_id", "Salesforce/blip-image-captioning-base"),
            device=self.device,
        )
        
        self.logger.info("Models initialized successfully")
    
    def generate_images_from_text(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompts.
        
        Args:
            prompts: List of text prompts.
            num_images_per_prompt: Number of images per prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated images.
        """
        self.logger.info(f"Generating images for {len(prompts)} prompts")
        
        all_images = []
        for prompt in prompts:
            images = self.text_to_image.generate(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                **kwargs
            )
            all_images.extend(images)
        
        self.logger.info(f"Generated {len(all_images)} images total")
        return all_images
    
    def generate_text_from_images(
        self,
        images: List[Image.Image],
        style: str = "detailed"
    ) -> List[str]:
        """Generate text descriptions from images.
        
        Args:
            images: List of PIL Images.
            style: Style of description ("detailed", "simple", "artistic").
            
        Returns:
            List of generated descriptions.
        """
        self.logger.info(f"Generating text for {len(images)} images")
        
        descriptions = []
        for image in images:
            description = self.image_to_text.generate_descriptive_caption(
                image, style=style
            )
            descriptions.append(description)
        
        self.logger.info(f"Generated {len(descriptions)} descriptions")
        return descriptions
    
    def evaluate_generation(
        self,
        generated_images: List[Image.Image],
        prompts: List[str],
        real_images: Optional[List[Image.Image]] = None,
    ) -> dict:
        """Evaluate generated images.
        
        Args:
            generated_images: List of generated images.
            prompts: List of prompts used for generation.
            real_images: Optional list of real images for FID.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        self.logger.info("Evaluating generated images...")
        
        results = self.evaluator.evaluate_generation(
            generated_images=generated_images,
            prompts=prompts,
            real_images=real_images,
        )
        
        self.logger.info("Evaluation completed")
        return results
    
    def run_demo(self) -> None:
        """Run a demonstration of the multimodal generative models."""
        self.logger.info("Running demo...")
        
        # Sample prompts
        demo_prompts = [
            "A beautiful sunset over the ocean with waves crashing on the shore",
            "A cute cat sitting on a windowsill looking outside",
            "A modern city skyline at night with bright lights",
            "A peaceful forest with tall pine trees and sunlight filtering through",
            "A colorful garden with blooming flowers in spring",
        ]
        
        # Generate images
        self.logger.info("Generating images from text prompts...")
        generated_images = self.generate_images_from_text(
            demo_prompts,
            num_images_per_prompt=1,
            num_inference_steps=20,
            guidance_scale=7.5,
        )
        
        # Generate descriptions
        self.logger.info("Generating text descriptions from images...")
        descriptions = self.generate_text_from_images(
            generated_images,
            style="detailed"
        )
        
        # Evaluate generation
        self.logger.info("Evaluating generation quality...")
        evaluation_results = self.evaluate_generation(
            generated_images=generated_images,
            prompts=demo_prompts,
        )
        
        # Print results
        self.logger.info("Demo Results:")
        self.logger.info(f"Generated {len(generated_images)} images")
        self.logger.info(f"Generated {len(descriptions)} descriptions")
        
        self.logger.info("Evaluation Metrics:")
        for metric, value in evaluation_results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Save results
        output_dir = Path(self.config.get("paths.output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        for i, (image, prompt) in enumerate(zip(generated_images, demo_prompts)):
            image_path = output_dir / f"generated_{i:03d}.png"
            image.save(image_path)
            self.logger.info(f"Saved image: {image_path}")
        
        # Save evaluation results
        import json
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        self.logger.info(f"Saved evaluation results: {results_path}")
        
        self.logger.info("Demo completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal Generative Models Application"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration"
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample dataset"
    )
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_dataset("data")
        return
    
    # Initialize application
    app = MultimodalGenerativeApp(config_path=args.config)
    
    # Run demo if requested
    if args.demo:
        app.run_demo()
    else:
        print("Use --demo to run a demonstration")
        print("Use --create-sample-data to create sample dataset")


if __name__ == "__main__":
    main()
