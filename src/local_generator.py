"""
Local Stable Diffusion Generator for Generative LexiArt
Provides completely free image generation without API limitations
"""

import torch
from PIL import Image
import streamlit as st
from typing import Optional, Dict, Any
import time

class LocalGenerator:
    """
    Local Stable Diffusion generator using Diffusers library
    Completely free with no API limitations
    """
    
    def __init__(self):
        """Initialize local generator"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_loaded = False
        
    def load_model(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Load Stable Diffusion model locally
        
        Args:
            model_id (str): Hugging Face model ID
        """
        try:
            if not self.model_loaded:
                with st.spinner(f"🔄 Loading model {model_id} locally (one-time setup)..."):
                    from diffusers import StableDiffusionPipeline
                    import torch
                    
                    # Test basic torch operations first
                    try:
                        test_tensor = torch.randn(1, 3, 32, 32)
                        st.write("✅ PyTorch operations working")
                    except Exception as e:
                        st.error(f"❌ PyTorch DLL error: {str(e)}")
                        st.info("Install Visual C++ Redistributable and restart")
                        return False
                    
                    # Load model with Windows-friendly settings
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,  # More reliable format
                        low_cpu_mem_usage=True  # Better for Windows
                    )
                    
                    self.pipe = self.pipe.to(self.device)
                    
                    # Windows-specific optimizations
                    if hasattr(self.pipe, 'enable_attention_slicing'):
                        self.pipe.enable_attention_slicing()
                        st.write("✅ Memory optimization enabled")
                    
                    # Test generation capability
                    try:
                        # Small test generation
                        with torch.no_grad():
                            test_result = self.pipe(
                                "test", 
                                num_inference_steps=1, 
                                width=64, 
                                height=64,
                                output_type="pil"
                            )
                        st.write("✅ Test generation successful")
                    except Exception as e:
                        st.error(f"❌ Generation test failed: {str(e)}")
                        return False
                    
                    self.model_loaded = True
                    st.success("✅ Local model loaded successfully!")
                    
        except ImportError as e:
            st.error(f"❌ Import error: {str(e)}")
            st.info("Run: pip install diffusers accelerate torch --index-url https://download.pytorch.org/whl/cpu")
            return False
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.info("Try restarting after installing Visual C++ Redistributable")
            return False
            
        return True
    
    def generate_image_local(self, prompt: str, negative_prompt: str = "",
                           width: int = 512, height: int = 512,
                           num_inference_steps: int = 20,
                           guidance_scale: float = 7.5,
                           seed: Optional[int] = None) -> Optional[Image.Image]:
        """
        Generate image using local Stable Diffusion
        
        Args:
            prompt (str): Generation prompt
            negative_prompt (str): Negative prompt
            width (int): Image width
            height (int): Image height
            num_inference_steps (int): Number of steps
            guidance_scale (float): Guidance scale
            seed (int, optional): Random seed
            
        Returns:
            PIL.Image.Image or None: Generated image
        """
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        try:
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
                )
            
            return result.images[0]
            
        except Exception as e:
            st.error(f"❌ Local generation error: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if local generation is available
        
        Returns:
            bool: True if can generate locally
        """
        try:
            import diffusers
            return True
        except ImportError:
            return False
    
    def get_device_info(self) -> str:
        """
        Get device information
        
        Returns:
            str: Device info
        """
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name()})"
        else:
            return "CPU"