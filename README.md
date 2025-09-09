# Generative LexiArt

A freemium AI image generation web application built with Streamlit that allows users to create high-quality images using AI models. Users get 5 free generations, then can upgrade to premium with a free HuggingFace API key for unlimited access.

view demo: https://lexiart.streamlit.app/

## Features

### Free Tier
- 5 free image generations per session
- 1024×1024 high-quality output
- Pollinations AI provider
- Prompt enhancement system
- Multiple download formats (original, phone wallpaper, desktop wallpaper)
- Intelligent caching system

### Premium Tier
- Unlimited image generations
- HuggingFace Stable Diffusion XL model
- Advanced generation parameters
- Priority processing
- Professional quality output
- Enhanced prompt processing

## Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/generative-lexiart.git
cd generative-lexiart
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

### Requirements

Create a `requirements.txt` file with:
```
streamlit>=1.28.0
requests>=2.31.0
Pillow>=10.0.0
huggingface-hub>=0.17.0
```

## Usage

### Getting Started

1. **Launch the app** and enter a descriptive prompt
2. **Choose enhancement level** (light, medium, heavy)
3. **Adjust advanced settings** if needed
4. **Generate your image** and download in multiple formats

### Upgrading to Premium

1. Visit [HuggingFace](https://huggingface.co/settings/tokens)
2. Create a free account
3. Generate an access token with "Read" permissions
4. Enter the token in the app's premium setup
5. Enjoy unlimited generations

### Writing Better Prompts

- Be specific and descriptive
- Include art style (oil painting, digital art, photography)
- Mention lighting and mood
- Add quality keywords (detailed, high quality, professional)

**Examples:**
- "Sunset over mountains, oil painting style, warm colors, detailed"
- "Cyberpunk city street, neon lights, futuristic, detailed digital art"
- "Portrait of a cat, professional photography, studio lighting"

## Configuration

### Environment Variables

Optional environment variables:
```bash
HUGGINGFACE_API_KEY=your_api_key_here  # For automatic premium setup
```

### File Structure

```
generative-lexiart/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── src/                           # Source code modules
│   ├── sessions_manager.py        # User session management
│   ├── freemium_ui_components.py  # UI components
│   ├── freemium_image_generator.py # Image generation logic
│   ├── image_generator.py         # HuggingFace integration
│   ├── pollinations_provider.py   # Free tier provider
│   ├── prompt_processor.py        # Prompt enhancement
│   ├── cache_manager.py           # Caching system
│   ├── pipeline.py                # Generation pipeline
│   └── utils.py                   # Utility functions
├── static/                        # Static files (auto-created)
│   ├── cache.json                 # Generation cache
│   ├── sessions.json              # User sessions
│   ├── visits.json                # Visit counter
│   └── generated_images/          # Cached images
└── README.md
```

## Deployment

### Streamlit Cloud

1. **Push to GitHub** with all files
2. **Connect to Streamlit Cloud**
3. **Deploy from your repository**
4. **Set Python version** to 3.9+ in advanced settings

### Local Production

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## Architecture

### Core Components

- **SessionManager**: Tracks user usage and premium status
- **FreemiumGenerator**: Routes between free and premium providers
- **PromptProcessor**: Enhances user prompts for better results
- **CacheManager**: Stores generated images to improve performance
- **PollinationsProvider**: Free tier image generation
- **ImageGenerator**: Premium HuggingFace integration

### Data Flow

1. User enters prompt
2. Session manager checks generation limits
3. Prompt processor enhances the input
4. Cache manager checks for existing images
5. Appropriate provider generates new image
6. Result is cached and displayed

## Troubleshooting

### Common Issues

**"Free generation limit reached"**
- Use the sidebar premium setup to get unlimited access
- Requires free HuggingFace account and API token

**"Hugging Face unavailable, using local generation"**
- Check API key validity
- Verify HuggingFace service status
- Local generation requires additional dependencies

**Import errors during deployment**
- Ensure `requirements.txt` is present and complete
- Check Python version compatibility
- Verify all source files are uploaded

**Duplicate button key errors**
- Each UI component uses unique keys
- Clear browser cache and refresh
- Check for multiple instances of the same component

### Memory Issues

If experiencing memory problems:
- Reduce cache size in `cache_manager.py`
- Clear cache periodically
- Use lighter dependencies for cloud deployment

## API Integration

### HuggingFace Setup

1. Create account at [huggingface.co](https://huggingface.co)
2. Navigate to Settings → Access Tokens
3. Generate token with "Read" permission
4. Copy token starting with "hf_"
5. Enter in app's premium setup

### Supported Models

- **Free Tier**: Pollinations AI (various models)
- **Premium**: Stable Diffusion XL Base 1.0

## Performance

### Caching System

- Intelligent caching based on prompt and parameters
- Automatic cleanup of old entries
- Significant performance improvement (up to 60%)
- File-based storage with JSON metadata

### Optimization Features

- Prompt enhancement for better results
- Parameter validation and optimization
- Progressive loading with status updates
- Memory-efficient image handling

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test locally
4. Commit with clear messages
5. Push and create pull request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic
- Test locally before committing

### Adding Features

- New providers go in separate files
- UI components use unique keys
- Add error handling and validation
- Update documentation

## License

MIT License - see LICENSE file for details

## Support

- Create GitHub issues for bugs
- Check existing issues before reporting
- Provide error logs and steps to reproduce
- Include system information (OS, Python version)

## Acknowledgments

- Streamlit for the web framework
- HuggingFace for AI model access
- Pollinations for free tier generation
- Pillow for image processing
