"""
Configuration settings for Generative LexiArt
Handles API keys, model settings, and application configuration
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings and configuration management"""
    
    def __init__(self):
        """Initialize settings from environment variables"""
        # API Configuration
        self.huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN', '')
        self.replicate_token = os.getenv('REPLICATE_API_TOKEN', '')
        self.together_api_key = os.getenv('TOGETHER_API_KEY', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Application Settings
        self.app_title = os.getenv('APP_TITLE', 'Generative LexiArt')
        self.default_model = os.getenv('DEFAULT_MODEL', 'CompVis/stable-diffusion-v1-4')
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE', '100'))
        self.image_quality = os.getenv('IMAGE_QUALITY', 'high')
        
        # File Paths
        self.generated_images_path = 'static/generated_images'
        self.cache_file_path = 'static/cache.json'
        
        # API URLs
        self.huggingface_api_url = "https://api-inference.huggingface.co/models"
        
        # Image Generation Settings
        self.default_width = 512
        self.default_height = 512
        self.default_num_inference_steps = 20
        self.default_guidance_scale = 7.5
        
        # Performance Settings
        self.enable_caching = True
        self.max_retry_attempts = 3
        self.api_timeout = 60  # seconds

    def get_api_headers(self, api_type: str = 'huggingface') -> Dict[str, str]:
        """
        Get API headers for different services
        
        Args:
            api_type (str): Type of API ('huggingface', 'replicate', etc.)
            
        Returns:
            Dict[str, str]: Headers dictionary for API requests
        """
        headers = {}
        
        if api_type == 'huggingface':
            if self.huggingface_token:
                headers['Authorization'] = f'Bearer {self.huggingface_token}'
            headers['Content-Type'] = 'application/json'
            
        elif api_type == 'replicate':
            if self.replicate_token:
                headers['Authorization'] = f'Token {self.replicate_token}'
            headers['Content-Type'] = 'application/json'
            
        elif api_type == 'together':
            if self.together_api_key:
                headers['Authorization'] = f'Bearer {self.together_api_key}'
            headers['Content-Type'] = 'application/json'
            
        return headers

    def get_model_url(self, model_name: Optional[str] = None) -> str:
        """
        Get full API URL for a specific model
        
        Args:
            model_name (str, optional): Name of the model to use
            
        Returns:
            str: Full API URL for the model
        """
        model = model_name or self.default_model
        return f"{self.huggingface_api_url}/{model}"

    def validate_configuration(self) -> bool:
        """
        Validate that required configuration is present
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check if at least one API token is configured
        api_tokens = [
            self.huggingface_token,
            self.replicate_token,
            self.together_api_key,
            self.openai_api_key
        ]
        
        if not any(api_tokens):
            return False
            
        # Check if required directories exist, create if not
        os.makedirs(self.generated_images_path, exist_ok=True)
        
        return True

def get_settings() -> Settings:
    """
    Get application settings instance
    
    Returns:
        Settings: Application settings object
    """
    return Settings()

def validate_api_key(api_type: str = 'huggingface') -> bool:
    """
    Validate if API key is configured for specified service
    
    Args:
        api_type (str): Type of API to validate
        
    Returns:
        bool: True if API key is configured, False otherwise
    """
    settings = get_settings()
    
    if api_type == 'huggingface':
        return bool(settings.huggingface_token)
    elif api_type == 'replicate':
        return bool(settings.replicate_token)
    elif api_type == 'together':
        return bool(settings.together_api_key)
    elif api_type == 'openai':
        return bool(settings.openai_api_key)
    
    return False