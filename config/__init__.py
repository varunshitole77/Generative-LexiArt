"""
Configuration module for Generative LexiArt
Simplified imports focusing on Hugging Face integration
"""

from .settings import (
    get_settings,
    is_configured,
    get_hf_token,
    get_generation_params,
    display_config_status,
    Settings
)

# For backward compatibility
def validate_api_key() -> bool:
    """Validate API key configuration (backward compatibility)"""
    return get_settings().validate_configuration()

__all__ = [
    'get_settings',
    'is_configured', 
    'get_hf_token',
    'get_generation_params',
    'display_config_status',
    'validate_api_key',
    'Settings'
]