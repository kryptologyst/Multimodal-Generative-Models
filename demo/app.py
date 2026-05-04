"""Streamlit demo application for multimodal generative models."""

import io
import random
from pathlib import Path
from typing import List, Optional

import streamlit as st
import torch
from PIL import Image

from src.app import MultimodalGenerativeApp
from src.utils.config import Config
from src.utils.device import get_device_info


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multimodal Generative Models",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🎨 Multimodal Generative Models")
    st.markdown("""
    This application demonstrates text-to-image and image-to-text generation using 
    state-of-the-art multimodal models including Stable Diffusion and CLIP/BLIP.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Device info
        device_info = get_device_info()
        st.info(f"**Device:** {device_info['device']}")
        
        # Safety settings
        st.subheader("🛡️ Safety Settings")
        nsfw_filter = st.checkbox("NSFW Filter", value=True)
        content_filter = st.checkbox("Content Filter", value=True)
        watermark = st.checkbox("Watermark Generated Images", value=True)
        
        # Generation parameters
        st.subheader("🎛️ Generation Parameters")
        num_inference_steps = st.slider("Inference Steps", 10, 50, 20)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
        num_images = st.slider("Images per Prompt", 1, 4, 1)
        
        # Image dimensions
        st.subheader("📐 Image Dimensions")
        width = st.selectbox("Width", [256, 512, 768, 1024], index=1)
        height = st.selectbox("Height", [256, 512, 768, 1024], index=1)
    
    # Initialize app (with caching)
    @st.cache_resource
    def load_app():
        """Load the multimodal app with caching."""
        return MultimodalGenerativeApp()
    
    app = load_app()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Text-to-Image", 
        "📝 Image-to-Text", 
        "🔄 Round-trip", 
        "📊 Evaluation"
    ])
    
    with tab1:
        st.header("Text-to-Image Generation")
        st.markdown("Generate images from text descriptions using Stable Diffusion.")
        
        # Text input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "Enter your prompt:",
                value="A beautiful sunset over the ocean with waves crashing on the shore",
                height=100,
                help="Describe the image you want to generate"
            )
            
            negative_prompt = st.text_area(
                "Negative prompt (optional):",
                value="blurry, low quality, distorted, ugly",
                height=60,
                help="Describe what you want to avoid in the generated image"
            )
        
        with col2:
            seed = st.number_input("Seed", value=42, help="Random seed for reproducibility")
            
            if st.button("🎨 Generate Image", type="primary"):
                if prompt.strip():
                    with st.spinner("Generating image..."):
                        try:
                            # Generate images
                            images = app.generate_images_from_text(
                                prompts=[prompt],
                                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                height=height,
                                width=width,
                                num_images_per_prompt=num_images,
                                seed=seed,
                            )
                            
                            # Display images
                            for i, image in enumerate(images):
                                st.image(image, caption=f"Generated Image {i+1}")
                                
                                # Download button
                                img_buffer = io.BytesIO()
                                image.save(img_buffer, format="PNG")
                                st.download_button(
                                    label=f"Download Image {i+1}",
                                    data=img_buffer.getvalue(),
                                    file_name=f"generated_image_{i+1}.png",
                                    mime="image/png"
                                )
                        
                        except Exception as e:
                            st.error(f"Error generating image: {str(e)}")
                else:
                    st.warning("Please enter a prompt!")
    
    with tab2:
        st.header("Image-to-Text Generation")
        st.markdown("Generate text descriptions from images using CLIP and BLIP.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to generate a text description"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
            # Description style
            style = st.selectbox(
                "Description Style:",
                ["detailed", "simple", "artistic"],
                help="Choose the style of description"
            )
            
            if st.button("📝 Generate Description", type="primary"):
                with st.spinner("Generating description..."):
                    try:
                        descriptions = app.generate_text_from_images(
                            images=[image],
                            style=style
                        )
                        
                        st.success("Generated Description:")
                        st.write(descriptions[0])
                        
                        # Copy to clipboard
                        st.code(descriptions[0])
                    
                    except Exception as e:
                        st.error(f"Error generating description: {str(e)}")
    
    with tab3:
        st.header("Round-trip Generation")
        st.markdown("Generate an image from text, then generate text from that image.")
        
        roundtrip_prompt = st.text_area(
            "Enter your prompt for round-trip generation:",
            value="A peaceful forest with tall pine trees and sunlight filtering through",
            height=100
        )
        
        if st.button("🔄 Start Round-trip", type="primary"):
            if roundtrip_prompt.strip():
                with st.spinner("Performing round-trip generation..."):
                    try:
                        # Step 1: Generate image from text
                        st.subheader("Step 1: Text → Image")
                        images = app.generate_images_from_text(
                            prompts=[roundtrip_prompt],
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            seed=seed,
                        )
                        
                        generated_image = images[0]
                        st.image(generated_image, caption="Generated Image")
                        
                        # Step 2: Generate text from image
                        st.subheader("Step 2: Image → Text")
                        descriptions = app.generate_text_from_images(
                            images=[generated_image],
                            style="detailed"
                        )
                        
                        st.success("Round-trip Description:")
                        st.write(descriptions[0])
                        
                        # Compare prompts
                        st.subheader("Comparison")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Prompt:**")
                            st.write(roundtrip_prompt)
                        
                        with col2:
                            st.write("**Generated Description:**")
                            st.write(descriptions[0])
                    
                    except Exception as e:
                        st.error(f"Error in round-trip generation: {str(e)}")
            else:
                st.warning("Please enter a prompt!")
    
    with tab4:
        st.header("Evaluation Metrics")
        st.markdown("Evaluate the quality of generated images using various metrics.")
        
        # Sample evaluation
        if st.button("📊 Run Sample Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    # Sample prompts for evaluation
                    eval_prompts = [
                        "A beautiful sunset over the ocean",
                        "A cat sitting on a windowsill",
                        "A modern city skyline at night",
                        "A peaceful forest with tall trees",
                        "A colorful garden with flowers",
                    ]
                    
                    # Generate images
                    images = app.generate_images_from_text(
                        prompts=eval_prompts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        seed=seed,
                    )
                    
                    # Evaluate
                    results = app.evaluate_generation(
                        generated_images=images,
                        prompts=eval_prompts,
                    )
                    
                    # Display results
                    st.success("Evaluation Results:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("CLIP Score (Mean)", f"{results['clip_score_mean']:.4f}")
                        st.metric("CLIP Score (Std)", f"{results['clip_score_std']:.4f}")
                        st.metric("Aesthetic Score", f"{results['aesthetic_score_mean']:.4f}")
                    
                    with col2:
                        st.metric("Diversity Score", f"{results['diversity_score']:.4f}")
                        st.metric("Aesthetic Std", f"{results['aesthetic_score_std']:.4f}")
                        st.metric("Min CLIP Score", f"{results['clip_score_min']:.4f}")
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    st.json(results)
                
                except Exception as e:
                    st.error(f"Error in evaluation: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Important Disclaimers
    
    **Research/Educational Use Only:** This application is intended for research and educational purposes only.
    
    **Generated Content:** All generated images and text are created by AI models and may not be accurate or appropriate for all contexts.
    
    **Safety:** While safety filters are enabled, users should exercise caution when generating content.
    
    **Performance:** Generation quality and speed depend on your hardware configuration.
    
    **Privacy:** Images uploaded for processing are not stored permanently.
    """)
    
    st.markdown("""
    ### 🔗 Resources
    
    - [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [CLIP Model](https://huggingface.co/openai/clip-vit-base-patch32)
    - [BLIP Model](https://huggingface.co/Salesforce/blip-image-captioning-base)
    - [Project Repository](https://github.com/kryptologyst)
    """)


if __name__ == "__main__":
    main()
