

from typing import Optional, Dict, Any, List
from PIL import Image
import streamlit as st
import time

class FreemiumImageGenerator:
    """
    Wrapper that handles both free (Pollinations) and premium (HuggingFace) generation
    """
    
    def __init__(self, session_manager):
        """Initialize with session manager"""
        self.session_manager = session_manager
        
        # Initialize providers
        from .pollinations_provider import PollinationsProvider
        self.pollinations = PollinationsProvider()
        
        # Initialize your existing HuggingFace generator (without parameters)
        from .image_generator import ImageGenerator
        self.huggingface_generator = ImageGenerator()  # Use default initialization
        
        # Premium model - ONLY ONE MODEL NOW
        self.default_model = "stabilityai/stable-diffusion-xl-base-1.0"
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get models available to current user - now returns empty since we hide selection"""
        # Not used anymore since we hide model selection
        return []
    
    def generate_image(self, prompt: str, model: Optional[str] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Generate image using appropriate provider
        
        Returns:
            tuple: (image, generation_time) or (None, 0) if failed
        """
        start_time = time.time()
        
        provider = self.session_manager.get_provider_for_generation()
        
        if provider == 'limit_reached':
            st.error("Free generation limit reached!")
            return None, 0
        
        elif provider == 'pollinations':
            # Free generation with Pollinations
            remaining = self.session_manager.get_remaining_free()
            if remaining > 1:
                st.info(f"Generating your image... ({remaining-1} free generations remaining)")
            else:
                st.info("Generating your last free image...")
            
            image = self.pollinations.generate_image(prompt, parameters)
            
            if image:
                self.session_manager.increment_usage()
                remaining = self.session_manager.get_remaining_free()
                if remaining <= 2:
                    st.success("Image generated! Running low on free generations.")
                else:
                    st.success("Image generated successfully!")
            
        elif provider == 'huggingface':
            # Premium generation using the single default model
            image = self._generate_with_existing_generator(prompt, self.default_model, parameters)
            
            if image:
                st.success("Premium image generated!")
        
        else:
            st.error("Unknown provider error")
            return None, 0
        
        generation_time = time.time() - start_time
        return image, generation_time
    
    def _generate_with_existing_generator(self, prompt: str, model: Optional[str] = None,
                                        parameters: Optional[Dict[str, Any]] = None) -> Optional[Image.Image]:
        """Generate using your existing HuggingFace generator"""
        try:
            # Always use the default model
            model = self.default_model
            
            # Use your existing generator's generate_image method
            result = self.huggingface_generator.generate_image(prompt, model, parameters)
            
            # Handle different return types from your existing generator
            if isinstance(result, tuple) and len(result) == 2:
                # If it returns (image, generation_time)
                image, gen_time = result
                return image
            elif result is not None:
                # If it just returns the image
                return result
            else:
                return None
                
        except Exception as e:
            st.error(f"HuggingFace generation error: {str(e)}")
            return None
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize generation parameters"""
        validated = {
            'width': max(256, min(1024, parameters.get('width', 512))),
            'height': max(256, min(1024, parameters.get('height', 512))),
            'num_inference_steps': max(10, min(100, parameters.get('num_inference_steps', 20))),
            'guidance_scale': max(1.0, min(20.0, parameters.get('guidance_scale', 7.5))),
            'negative_prompt': parameters.get('negative_prompt', ''),
            'seed': parameters.get('seed', -1)
        }
        
        # Ensure dimensions are multiples of 8
        validated['width'] = (validated['width'] // 8) * 8
        validated['height'] = (validated['height'] // 8) * 8
        
        return validated
    
    def get_generation_info(self) -> Dict[str, Any]:
        """Get information about current generation mode"""
        if self.session_manager.is_premium_user():
            return {
                'mode': 'premium',
                'provider': 'HuggingFace',
                'unlimited': True,
                'quality': 'High'
            }
        else:
            return {
                'mode': 'free',
                'provider': 'AI Generator',
                'remaining': self.session_manager.get_remaining_free(),
                'unlimited': False,
                'quality': 'Standard'
            }
    
    def estimate_generation_time(self, parameters: Dict[str, Any]) -> float:
        """Estimate generation time based on current provider"""
        if self.session_manager.is_premium_user():
            # HuggingFace time estimate
            base_time = 15
            width = parameters.get('width', 512)
            height = parameters.get('height', 512)
            steps = parameters.get('num_inference_steps', 20)
            
            resolution_factor = (width * height) / (512 * 512)
            steps_factor = steps / 20
            
            return base_time * resolution_factor * steps_factor
        else:
            # Pollinations time estimate
            return 10.0