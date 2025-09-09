import requests
import time
import os
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import io
import base64
import streamlit as st

class ImageGenerator:
    """
    Streamlined image generation class using only Hugging Face
    Features intelligent fallback and local generation
    """
    
    def __init__(self):
        """Initialize Hugging Face-only image generator"""
        # Direct settings instead of importing config
        self.huggingface_token = os.getenv('HUGGINGFACE_API_KEY') or st.session_state.get('huggingface_api_key', '')
        
        # Single Stability AI model
        self.default_model = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Initialize local generator as fallback - FIX: Always initialize these attributes
        self.local_generator = None
        self.use_local = False
        self.local_device = "cpu"  # FIX: Always initialize local_device
        
        self._initialize_generation_method()

    def _initialize_generation_method(self):
        """Initialize the best available generation method"""
        # First try: Hugging Face Inference Client
        if self._test_hf_inference_client():
            self.generation_method = "inference_client"
            return
        
        # Second try: Local generation with diffusers
        if self._initialize_local_generator():
            self.generation_method = "local"
            self.use_local = True
            return
        
        # Fallback: Direct API calls
        self.generation_method = "api_fallback"

    def _test_hf_inference_client(self) -> bool:
        """Test if Hugging Face Inference Client is available and working"""
        try:
            from huggingface_hub import InferenceClient
            
            if not self.huggingface_token:
                return False
            
            # Quick test - just check if client initializes
            client = InferenceClient(token=self.huggingface_token)
            return True
            
        except ImportError:
            st.warning("Install huggingface_hub: pip install huggingface_hub")
            return False
        except Exception:
            return False

    def _initialize_local_generator(self) -> bool:
        """Initialize local Stable Diffusion generator"""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            # Test if dependencies are available and set device
            if torch.cuda.is_available():
                self.local_device = "cuda"
            else:
                self.local_device = "cpu"
            
            # Test basic torch operations
            test_tensor = torch.randn(1, 3, 32, 32)
            return True
            
        except ImportError:
            return False
        except Exception:
            return False

    def _load_local_model(self) -> bool:
        """Load local model on demand"""
        if self.local_generator is not None:
            return True
        
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            model_id = "runwayml/stable-diffusion-v1-5"
            
            with st.spinner("Loading local model (one-time setup)..."):
                self.local_generator = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.local_device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    variant="fp16" if self.local_device == "cuda" else None
                )
                
                # FIX: Ensure local_device is always set before using it
                if not hasattr(self, 'local_device'):
                    self.local_device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.local_generator = self.local_generator.to(self.local_device)
                
                # Optimize for memory
                if hasattr(self.local_generator, 'enable_attention_slicing'):
                    self.local_generator.enable_attention_slicing()
                
                if self.local_device == "cuda" and hasattr(self.local_generator, 'enable_memory_efficient_attention'):
                    self.local_generator.enable_memory_efficient_attention()
            
            return True
            
        except ImportError as e:
            st.error(f"Missing dependencies: {str(e)}")
            st.info("Install: pip install diffusers transformers accelerate")
            return False
        except Exception as e:
            st.error(f"Failed to load local model: {str(e)}")
            st.info("Try: pip install --upgrade diffusers torch huggingface_hub")
            return False

    def generate_image_local(self, prompt: str, parameters: Dict[str, Any]) -> Tuple[Optional[Image.Image], float]:
        """Generate image using local Stable Diffusion"""
        if not self._load_local_model():
            return None, 0.0
        
        try:
            start_time = time.time()
            
            # Extract parameters
            width = parameters.get('width', 1024)
            height = parameters.get('height', 1024)
            num_steps = parameters.get('num_inference_steps', 25)
            guidance_scale = parameters.get('guidance_scale', 7.5)
            negative_prompt = parameters.get('negative_prompt', '')
            
            # FIX: Check if local_generator was loaded successfully
            if self.local_generator is None:
                st.error("Local model not loaded")
                return None, 0.0
            
            # Generate image
            with st.spinner("Generating with local model..."):
                result = self.local_generator(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale
                )
            
            generation_time = time.time() - start_time
            return result.images[0], generation_time
            
        except Exception as e:
            generation_time = time.time() - start_time if 'start_time' in locals() else 0.0
            st.error(f"Local generation error: {str(e)}")
            return None, generation_time

    def generate_image_inference_client(self, prompt: str, parameters: Dict[str, Any]) -> Tuple[Optional[Image.Image], float]:
        """Generate image using Hugging Face Inference Client"""
        try:
            from huggingface_hub import InferenceClient
            
            client = InferenceClient(token=self.huggingface_token)
            start_time = time.time()
            
            # Try using the default SDXL model
            try:
                with st.spinner("Creating artwork with AI..."):
                    image = client.text_to_image(
                        prompt=prompt,
                        model=self.default_model,
                        width=parameters.get('width', 1024),
                        height=parameters.get('height', 1024)
                    )
                
                generation_time = time.time() - start_time
                return image, generation_time
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle quota exceeded
                if "quota" in error_msg or "limit" in error_msg:
                    st.warning("API quota exceeded, switching to local generation...")
                    self.use_local = True
                    return self.generate_image_local(prompt, parameters)
                
                # Handle model loading errors
                elif "loading" in error_msg or "503" in error_msg:
                    st.info("Model loading, please wait...")
                    time.sleep(5)
                    try:
                        image = client.text_to_image(prompt=prompt, model=self.default_model)
                        generation_time = time.time() - start_time
                        return image, generation_time
                    except:
                        return self.generate_image_local(prompt, parameters)
                
                else:
                    st.warning("Hugging Face unavailable, using local generation...")
                    return self.generate_image_local(prompt, parameters)
        
        except ImportError:
            st.error("Missing dependency: pip install huggingface_hub")
            return None, 0.0
        except Exception as e:
            st.error(f"Inference client error: {str(e)}")
            return self.generate_image_local(prompt, parameters)

    def generate_image_api_fallback(self, prompt: str, parameters: Dict[str, Any]) -> Tuple[Optional[Image.Image], float]:
        """Fallback API method using the single default model"""
        if not self.huggingface_token:
            st.error("Hugging Face token required")
            return None, 0.0
        
        # Use the single default model
        model = self.default_model
        
        try:
            start_time = time.time()
            
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {self.huggingface_token}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "width": parameters.get('width', 1024),
                    "height": parameters.get('height', 1024),
                    "num_inference_steps": parameters.get('num_inference_steps', 25),
                    "guidance_scale": parameters.get('guidance_scale', 7.5)
                }
            }
            
            if parameters.get('negative_prompt'):
                payload["parameters"]["negative_prompt"] = parameters['negative_prompt']
            
            with st.spinner(f"Generating with premium model..."):
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                generation_time = time.time() - start_time
                return image, generation_time
            
            elif response.status_code == 503:
                st.info(f"Model is loading, please wait...")
                time.sleep(5)
                # Retry once
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    generation_time = time.time() - start_time
                    return image, generation_time
            
            elif response.status_code == 429:
                st.warning("Rate limit reached, switching to local generation...")
                return self.generate_image_local(prompt, parameters)
            
        except Exception as e:
            st.warning(f"API error: {str(e)}, switching to local generation...")
        
        # If API fails, try local generation
        return self.generate_image_local(prompt, parameters)

    def generate_image(self, prompt: str, model: Optional[str] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Image.Image], float]:
        """Main generation method with intelligent routing"""
        # Validate and set default parameters
        params = self._validate_parameters(parameters or {})
        
        # Route to appropriate generation method
        if self.use_local or self.generation_method == "local":
            return self.generate_image_local(prompt, params)
        elif self.generation_method == "inference_client":
            return self.generate_image_inference_client(prompt, params)
        else:
            return self.generate_image_api_fallback(prompt, params)

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters"""
        return {
            'width': min(4096, max(256, parameters.get('width', 1024))),
            'height': min(4096, max(256, parameters.get('height', 1024))),
            'num_inference_steps': max(15, min(50, parameters.get('num_inference_steps', 25))),
            'guidance_scale': max(5.0, min(15.0, parameters.get('guidance_scale', 7.5))),
            'negative_prompt': parameters.get('negative_prompt', '')
        }

    def get_available_models(self) -> List[str]:
        """Get list of available models - now returns empty since selection is hidden"""
        return []

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get available providers for dashboard"""
        return [
            {
                'name': 'huggingface',
                'type': 'premium' if self.huggingface_token else 'free',
                'available': True,
                'free_tier': 'Unlimited with API key'
            }
        ]

    def test_connection(self) -> bool:
        """Test if generation is working"""
        try:
            if self.use_local:
                return self._initialize_local_generator()
            else:
                return self._test_hf_inference_client()
        except:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current generator status"""
        return {
            'method': self.generation_method,
            'local_available': self._initialize_local_generator(),
            'hf_client_available': self._test_hf_inference_client(),
            'current_model': "Premium AI Model",
            'quota_exceeded': False
        }

    def estimate_generation_time(self, parameters: Dict[str, Any]) -> float:
        """Estimate generation time based on method"""
        if self.use_local:
            return 45.0
        else:
            return 15.0

    def switch_to_local(self):
        """Force switch to local generation"""
        self.use_local = True
        self.generation_method = "local"
        # FIX: Ensure local_device is set when switching to local
        if not hasattr(self, 'local_device'):
            try:
                import torch
                self.local_device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                self.local_device = "cpu"
        st.info("Switched to local generation mode")
