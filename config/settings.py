"""
Enhanced Settings Configuration for Generative LexiArt
Includes timeout and error handling settings
"""

import os
from typing import Dict, Any
import streamlit as st

class Settings:
    """Application settings with timeout and error handling configuration"""
    
    def __init__(self):
        # API Configuration
        self.huggingface_token = os.getenv('HUGGINGFACE_API_KEY') or st.session_state.get('huggingface_api_key', '')
        self.replicate_api_token = os.getenv('REPLICATE_API_TOKEN', '')
        
        # Timeout Settings
        self.api_timeout = int(os.getenv('API_TIMEOUT', '45'))  # seconds
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '2'))  # seconds
        
        # File Paths
        self.cache_file_path = os.getenv('CACHE_FILE_PATH', 'static/cache.json')
        self.generated_images_path = os.getenv('GENERATED_IMAGES_PATH', 'static/generated_images')
        self.session_file_path = os.getenv('SESSION_FILE_PATH', 'static/sessions.json')
        
        # Cache Settings
        self.enable_caching = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE', '100'))
        
        # Generation Settings
        self.max_free_generations = int(os.getenv('MAX_FREE_GENERATIONS', '5'))
        self.default_width = int(os.getenv('DEFAULT_WIDTH', '512'))
        self.default_height = int(os.getenv('DEFAULT_HEIGHT', '512'))
        
        # Error Handling
        self.enable_fallback = os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true'
        self.show_debug_info = os.getenv('SHOW_DEBUG_INFO', 'false').lower() == 'true'
        
        # Service URLs
        self.pollinations_url = os.getenv('POLLINATIONS_URL', 'https://image.pollinations.ai/prompt')
        self.huggingface_api_url = os.getenv('HUGGINGFACE_API_URL', 'https://api-inference.huggingface.co/models')
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            os.path.dirname(self.cache_file_path),
            self.generated_images_path,
            os.path.dirname(self.session_file_path)
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    st.warning(f"Could not create directory {directory}: {str(e)}")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for requests"""
        return {
            'timeout': self.api_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'headers': {
                'User-Agent': 'Generative-LexiArt/1.0',
                'Accept': 'application/json, image/*',
                'Connection': 'keep-alive'
            }
        }
    
    def get_generation_limits(self) -> Dict[str, Any]:
        """Get generation limits and constraints"""
        return {
            'free_generations': self.max_free_generations,
            'max_width': 1024,
            'max_height': 1024,
            'min_width': 256,
            'min_height': 256,
            'max_steps': 50,
            'min_steps': 10,
            'max_guidance': 20.0,
            'min_guidance': 1.0,
            'max_prompt_length': 500
        }
    
    def validate_settings(self) -> Dict[str, Any]:
        """Validate current settings and return status"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check API keys
        if not self.huggingface_token:
            validation['warnings'].append("HuggingFace API key not configured - premium features unavailable")
        
        # Check timeouts
        if self.api_timeout < 30:
            validation['warnings'].append("API timeout is quite short - may cause frequent timeouts")
        elif self.api_timeout > 120:
            validation['warnings'].append("API timeout is very long - users may wait too long")
        
        # Check directories
        try:
            test_file = os.path.join(self.generated_images_path, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            validation['errors'].append(f"Cannot write to generated images directory: {str(e)}")
            validation['valid'] = False
        
        return validation
    
    def update_from_session(self):
        """Update settings from Streamlit session state"""
        if 'huggingface_api_key' in st.session_state:
            self.huggingface_token = st.session_state.huggingface_api_key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for logging/debugging"""
        return {
            'api_timeout': self.api_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enable_caching': self.enable_caching,
            'max_cache_size': self.max_cache_size,
            'max_free_generations': self.max_free_generations,
            'enable_fallback': self.enable_fallback,
            'show_debug_info': self.show_debug_info,
            'has_huggingface_token': bool(self.huggingface_token),
            'has_replicate_token': bool(self.replicate_api_token)
        }

def get_settings() -> Settings:
    """Get application settings (cached in session state)"""
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = Settings()
    
    # Update from session state
    st.session_state.app_settings.update_from_session()
    
    return st.session_state.app_settings

def show_settings_debug():
    """Show settings debug information (for troubleshooting)"""
    settings = get_settings()
    
    with st.expander("Debug: Current Settings"):
        st.json(settings.to_dict())
        
        validation = settings.validate_settings()
        
        if validation['valid']:
            st.success("Settings validation passed")
        else:
            st.error("Settings validation failed")
        
        if validation['warnings']:
            st.warning("Warnings: " + "; ".join(validation['warnings']))
        
        if validation['errors']:
            st.error("Errors: " + "; ".join(validation['errors']))