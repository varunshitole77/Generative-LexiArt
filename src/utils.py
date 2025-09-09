import os
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
import streamlit as st

def generate_filename(prompt: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique filename based on prompt and timestamp
    
    Args:
        prompt (str): The image generation prompt
        timestamp (datetime, optional): Timestamp for the filename
        
    Returns:
        str: Generated filename
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create a hash of the prompt for uniqueness
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    # Format timestamp
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"generated_{time_str}_{prompt_hash}.png"
    
    return filename

def save_image(image: Image.Image, filepath: str) -> bool:
    """
    Save PIL Image to specified filepath
    
    Args:
        image (PIL.Image.Image): Image to save
        filepath (str): Full path where to save the image
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save image
        image.save(filepath, 'PNG', optimize=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return False

def load_image(filepath: str) -> Optional[Image.Image]:
    """
    Load image from filepath
    
    Args:
        filepath (str): Path to the image file
        
    Returns:
        PIL.Image.Image or None: Loaded image or None if error
    """
    try:
        if os.path.exists(filepath):
            return Image.open(filepath)
        else:
            st.warning(f"Image file not found: {filepath}")
            return None
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def clean_prompt(prompt: str) -> str:
    """
    Clean and normalize the input prompt
    
    Args:
        prompt (str): Raw input prompt
        
    Returns:
        str: Cleaned prompt
    """
    # Remove extra whitespace and newlines
    cleaned = ' '.join(prompt.strip().split())
    
    # Remove any potentially problematic characters
    cleaned = cleaned.replace('\n', ' ').replace('\r', '')
    
    # Limit length to reasonable size
    max_length = 500
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + '...'
    
    return cleaned

def format_generation_time(seconds: float) -> str:
    """
    Format generation time for display
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def create_generation_metadata(prompt: str, model: str, parameters: Dict[str, Any], 
                              generation_time: float) -> Dict[str, Any]:
    """
    Create metadata dictionary for generated image
    
    Args:
        prompt (str): Generation prompt
        model (str): Model used
        parameters (Dict[str, Any]): Generation parameters
        generation_time (float): Time taken to generate
        
    Returns:
        Dict[str, Any]: Metadata dictionary
    """
    return {
        'prompt': prompt,
        'model': model,
        'parameters': parameters,
        'generation_time': generation_time,
        'timestamp': datetime.now().isoformat(),
        'formatted_time': format_generation_time(generation_time)
    }

def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load JSON file safely
    
    Args:
        filepath (str): Path to JSON file
        
    Returns:
        Dict[str, Any] or None: Loaded JSON data or None if error
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

def save_json_file(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save data to JSON file safely
    
    Args:
        data (Dict[str, Any]): Data to save
        filepath (str): Path to save the file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        st.error(f"Error saving JSON file: {str(e)}")
        return False

def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath (str): Path to file
        
    Returns:
        str: Formatted file size
    """
    try:
        if not os.path.exists(filepath):
            return "File not found"
            
        size_bytes = os.path.getsize(filepath)
        
        # Convert to appropriate unit
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
        
    except Exception as e:
        return f"Error: {str(e)}"

def validate_image_parameters(width: int, height: int, steps: int, 
                            guidance_scale: float) -> Dict[str, Any]:
    """
    Validate and normalize image generation parameters
    
    Args:
        width (int): Image width
        height (int): Image height
        steps (int): Number of inference steps
        guidance_scale (float): Guidance scale
        
    Returns:
        Dict[str, Any]: Validated parameters
    """
    # Validate and clamp values to reasonable ranges
    width = max(256, min(1024, width))
    height = max(256, min(1024, height))
    steps = max(10, min(100, steps))
    guidance_scale = max(1.0, min(20.0, guidance_scale))
    
    # Ensure dimensions are multiples of 8 (required for most diffusion models)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    return {
        'width': width,
        'height': height,
        'num_inference_steps': steps,
        'guidance_scale': guidance_scale
    }
