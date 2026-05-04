# Multimodal Generative Models

A research-ready implementation of multimodal generative models for text↔image generation and editing using Stable Diffusion, CLIP, and BLIP.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text descriptions using Stable Diffusion
- **Image-to-Text Generation**: Generate detailed captions from images using CLIP and BLIP models
- **Comprehensive Evaluation**: CLIP score, FID, aesthetic score, and diversity metrics
- **Interactive Demo**: Streamlit web application for easy experimentation
- **Production Ready**: Clean code, type hints, comprehensive testing, and safety features

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Multimodal-Generative-Models.git
cd Multimodal-Generative-Models

# Install dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .
```

### Basic Usage

```python
from src.app import MultimodalGenerativeApp

# Initialize the application
app = MultimodalGenerativeApp()

# Generate images from text
prompts = ["A beautiful sunset over the ocean"]
images = app.generate_images_from_text(prompts)

# Generate text from images
descriptions = app.generate_text_from_images(images)
```

### Command Line Interface

```bash
# Run demonstration
python -m src.app --demo

# Create sample dataset
python -m src.app --create-sample-data

# Run with custom config
python -m src.app --config configs/custom.yaml --demo
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── text_to_image.py
│   │   └── image_to_text.py
│   ├── data/              # Data handling
│   │   └── dataset.py
│   ├── eval/              # Evaluation metrics
│   │   └── metrics.py
│   ├── utils/             # Utilities
│   │   ├── config.py
│   │   ├── device.py
│   │   └── logging.py
│   └── app.py             # Main application
├── configs/               # Configuration files
│   └── default.yaml
├── demo/                  # Demo applications
│   └── app.py
├── data/                  # Data directory
├── tests/                 # Unit tests
├── assets/                # Generated assets
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Models

### Text-to-Image Generation

- **Stable Diffusion v1.5**: High-quality image generation from text prompts
- **Multiple Schedulers**: DPMSolverMultistep, EulerDiscrete, LMSDiscrete, PNDM
- **Customizable Parameters**: Inference steps, guidance scale, image dimensions
- **Safety Features**: NSFW filtering, content filtering, watermarking

### Image-to-Text Generation

- **CLIP**: Cross-modal similarity and retrieval
- **BLIP**: Image captioning and visual question answering
- **Multiple Styles**: Detailed, simple, and artistic descriptions
- **Similarity Scoring**: CLIP-based image-text matching

## Evaluation Metrics

- **CLIP Score**: Semantic similarity between generated images and prompts
- **FID (Fréchet Inception Distance)**: Quality assessment compared to real images
- **Aesthetic Score**: Visual appeal and composition quality
- **Diversity Score**: Variation among generated images

## Configuration

The application uses YAML configuration files for easy customization:

```yaml
model:
  text_to_image:
    model_id: "runwayml/stable-diffusion-v1-5"
    num_inference_steps: 20
    guidance_scale: 7.5
    height: 512
    width: 512

evaluation:
  metrics: ["clip_score", "fid", "aesthetic_score"]
  num_samples: 100

safety:
  nsfw_filter: true
  content_filter: true
  watermark: true
```

## Device Support

- **CUDA**: Full GPU acceleration for NVIDIA GPUs
- **MPS**: Apple Silicon GPU support (M1/M2/M3)
- **CPU**: Fallback CPU inference
- **Automatic Detection**: Automatically selects the best available device

## Safety and Ethics

### Important Disclaimers

- **Research/Educational Use Only**: This project is intended for research and educational purposes
- **Generated Content**: AI-generated images and text may not be accurate or appropriate for all contexts
- **Safety Filters**: Built-in NSFW and content filtering (can be disabled for research)
- **Watermarking**: Optional watermarking of generated content
- **Privacy**: Uploaded images are processed locally and not stored permanently

### Responsible Use Guidelines

1. **Content Generation**: Be mindful of the content you generate and its potential impact
2. **Bias Awareness**: AI models may exhibit biases present in training data
3. **Attribution**: Clearly indicate when content is AI-generated
4. **Legal Compliance**: Ensure compliance with local laws and regulations
5. **Ethical Considerations**: Consider the ethical implications of generated content

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis
- **Testing**: Comprehensive unit tests

## Performance

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only inference
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX 4090 or similar

### Optimization Features

- **Mixed Precision**: Automatic mixed precision training
- **Memory Efficient Attention**: Reduced memory usage
- **Batch Processing**: Efficient batch generation
- **Model Caching**: Cached model loading for faster startup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) by Stability AI
- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) by OpenAI
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) by Salesforce
- [Hugging Face Transformers](https://huggingface.co/transformers/) for model implementations
- [Diffusers](https://huggingface.co/docs/diffusers/) for diffusion model pipelines

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multimodal_generative_models,
  title={Multimodal Generative Models: Text↔Image Generation and Editing},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multimodal-Generative-Models}
}
```

## Support

For questions, issues, or contributions, please:

1. Check the [Issues](https://github.com/kryptologyst/Multimodal-Generative-Models/issues) page
2. Create a new issue with detailed information
3. Follow the contributing guidelines

---

**Disclaimer**: This software is provided for research and educational purposes only. Users are responsible for ensuring their use complies with applicable laws and ethical guidelines.
# Multimodal-Generative-Models
