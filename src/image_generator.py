"""
Image Generator for Generative LexiArt
Handles API integration for text-to-image generation using diffusion models
"""

import requests
import time
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import base64
import streamlit as st
from config.settings import get_settings

class ImageGenerator:
    """
    Main image generation class supporting multiple APIs
    Currently configured for Hugging Face Inference API
    """
    
    def __init__(self, api_type: str = 'huggingface'):
        """
        Initialize image generator with specified API
        
        Args:
            api_type (str): Type of API to use ('huggingface', 'replicate', 'local')
        """
        self.api_type = api_type
        self.settings = get_settings()
        
        # Initialize local generator if needed
        if api_type == 'local':
            try:
                from .local_generator import LocalGenerator
                self.local_gen = LocalGenerator()
            except ImportError:
                st.error("❌ Local generation not available. Install: pip install diffusers accelerate")
                self.local_gen = None
        
        # Validate API configuration for non-local types
        if api_type != 'local' and not self.settings.validate_configuration():
            st.warning("⚠️ API configuration missing. Switching to local generation...")
            self.api_type = 'local'
        
        # Set up API-specific configurations
        self.setup_api_config()

    def setup_api_config(self) -> None:
        """Setup API-specific configuration"""
        
        if self.api_type == 'huggingface':
            self.api_url_base = self.settings.huggingface_api_url
            self.headers = self.settings.get_api_headers('huggingface')
            self.default_model = self.settings.default_model
            
        elif self.api_type == 'replicate':
            self.api_url_base = "https://api.replicate.com/v1/predictions"
            self.headers = self.settings.get_api_headers('replicate')
            self.default_model = "stability-ai/stable-diffusion"
            
        elif self.api_type == 'together':
            self.api_url_base = "https://api.together.xyz/v1/images/generations"
            self.headers = self.settings.get_api_headers('together')
            self.default_model = "runwayml/stable-diffusion-v1-5"
            
        elif self.api_type == 'local':
            self.api_url_base = None
            self.headers = {}
            self.default_model = "runwayml/stable-diffusion-v1-5"
            
        else:
            # Fallback for unknown API types
            self.api_url_base = None
            self.headers = {}
            self.default_model = "runwayml/stable-diffusion-v1-5"

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize generation parameters
        
        Args:
            parameters (Dict[str, Any]): Raw parameters
            
        Returns:
            Dict[str, Any]: Validated parameters
        """
        validated = {
            'width': max(256, min(1024, parameters.get('width', 512))),
            'height': max(256, min(1024, parameters.get('height', 512))),
            'num_inference_steps': max(10, min(100, parameters.get('num_inference_steps', 20))),
            'guidance_scale': max(1.0, min(20.0, parameters.get('guidance_scale', 7.5))),
            'negative_prompt': parameters.get('negative_prompt', ''),
            'seed': parameters.get('seed', -1)  # -1 for random
        }
        
        # Ensure dimensions are multiples of 8 (required for most diffusion models)
        validated['width'] = (validated['width'] // 8) * 8
        validated['height'] = (validated['height'] // 8) * 8
        
        return validated

    def generate_image_huggingface(self, prompt: str, model: Optional[str] = None, 
                                 parameters: Optional[Dict[str, Any]] = None) -> Optional[Image.Image]:
        """
        Generate image using Hugging Face Inference Client (Modern API)
        
        Args:
            prompt (str): Text prompt for generation
            model (str, optional): Model to use
            parameters (Dict[str, Any], optional): Generation parameters
            
        Returns:
            PIL.Image.Image or None: Generated image
        """
        try:
            # Import the modern InferenceClient
            from huggingface_hub import InferenceClient
            
            # Prepare parameters
            params = parameters or {}
            validated_params = self.validate_parameters(params)
            
            # Initialize InferenceClient with token
            client = InferenceClient(token=self.settings.huggingface_token)
            
            # Show generation progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Make API request with retries
            max_retries = self.settings.max_retry_attempts
            
            for attempt in range(max_retries):
                try:
                    status_text.text(f"🎨 Generating with Hugging Face... (Attempt {attempt + 1}/{max_retries})")
                    progress_bar.progress(0.3)
                    
                    # Use the working approach - let HF pick the best model automatically
                    # or try specific models that are known to work
                    if model and model != "auto":
                        st.write(f"🔍 Trying specific model: {model}")
                        try:
                            image = client.text_to_image(
                                prompt=prompt,
                                model=model,
                                **{k: v for k, v in validated_params.items() 
                                   if k in ['width', 'height', 'num_inference_steps', 'guidance_scale', 'negative_prompt'] and v is not None}
                            )
                        except Exception as model_error:
                            st.warning(f"⚠️ Model {model} failed, using auto-selection...")
                            image = client.text_to_image(prompt=prompt)
                    else:
                        # Let Hugging Face automatically select the best available model
                        st.write("🤖 Using automatic model selection (recommended)")
                        image = client.text_to_image(prompt=prompt)
                    
                    progress_bar.progress(0.9)
                    
                    if image:
                        progress_bar.progress(1.0)
                        status_text.text("✅ Image generated successfully!")
                        
                        # Clean up progress indicators after a short delay
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        return image
                    else:
                        st.error("❌ No image returned from API")
                        return None
                        
                except Exception as api_error:
                    error_msg = str(api_error)
                    
                    if "503" in error_msg or "loading" in error_msg.lower():
                        status_text.text("🔄 Model is loading, please wait...")
                        time.sleep(10)
                        progress_bar.progress(0.1)
                        continue
                    elif "404" in error_msg:
                        st.warning(f"⚠️ Model not found, using auto-selection...")
                        # Fallback to auto-selection
                        try:
                            image = client.text_to_image(prompt=prompt)
                            if image:
                                progress_bar.progress(1.0)
                                status_text.text("✅ Image generated with auto-selection!")
                                time.sleep(1)
                                progress_bar.empty()
                                status_text.empty()
                                return image
                        except:
                            pass
                        return None
                    elif "403" in error_msg:
                        st.error("❌ API permission error - check your token permissions")
                        return None
                    else:
                        st.warning(f"⚠️ Attempt {attempt + 1} error: {error_msg}")
                        
                        if attempt == max_retries - 1:
                            # Last attempt - try auto-selection as final fallback
                            try:
                                st.info("🔄 Final attempt with auto-selection...")
                                image = client.text_to_image(prompt=prompt)
                                if image:
                                    progress_bar.progress(1.0)
                                    status_text.text("✅ Image generated with auto-selection!")
                                    time.sleep(1)
                                    progress_bar.empty()
                                    status_text.empty()
                                    return image
                            except:
                                pass
                            
                            progress_bar.empty()
                            status_text.empty()
                            return None
                        
                        time.sleep(2)
            
            progress_bar.empty()
            status_text.empty()
            return None
            
        except ImportError:
            st.error("❌ Please install: pip install huggingface_hub")
            return None
        except Exception as e:
            st.error(f"Unexpected error during Hugging Face generation: {str(e)}")
            return None

    def generate_image_together(self, prompt: str, model: Optional[str] = None,
                              parameters: Optional[Dict[str, Any]] = None) -> Optional[Image.Image]:
        """
        Generate image using Together AI API
        
        Args:
            prompt (str): Text prompt for generation
            model (str, optional): Model to use
            parameters (Dict[str, Any], optional): Generation parameters
            
        Returns:
            PIL.Image.Image or None: Generated image
        """
        try:
            # Prepare parameters
            params = parameters or {}
            validated_params = self.validate_parameters(params)
            
            # Together AI payload format
            payload = {
                "model": "runwayml/stable-diffusion-v1-5",
                "prompt": prompt,
                "width": validated_params['width'],
                "height": validated_params['height'],
                "steps": validated_params['num_inference_steps'],
                "n": 1
            }
            
            if validated_params.get('negative_prompt'):
                payload["negative_prompt"] = validated_params['negative_prompt']
            
            # Show generation progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Make API request
            max_retries = self.settings.max_retry_attempts
            
            for attempt in range(max_retries):
                try:
                    status_text.text(f"🎨 Generating with Together AI... (Attempt {attempt + 1}/{max_retries})")
                    progress_bar.progress(0.3)
                    
                    response = requests.post(
                        self.api_url_base,
                        headers=self.headers,
                        json=payload,
                        timeout=self.settings.api_timeout
                    )
                    
                    progress_bar.progress(0.7)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'data' in result and result['data']:
                            # Together AI returns base64 encoded images
                            image_data = result['data'][0]['b64_json']
                            
                            # Decode base64 image
                            import base64
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Image generated successfully!")
                            
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                            
                            return image
                    
                    else:
                        error_msg = f"Together AI error {response.status_code}: {response.text}"
                        st.error(error_msg)
                        
                        if attempt == max_retries - 1:
                            progress_bar.empty()
                            status_text.empty()
                            return None
                        
                        time.sleep(2)
                        
                except requests.exceptions.Timeout:
                    st.warning(f"Request timeout on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        progress_bar.empty()
                        status_text.empty()
                        return None
                    time.sleep(3)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                    if attempt == max_retries - 1:
                        progress_bar.empty()
                        status_text.empty()
                        return None
                    time.sleep(2)
            
            progress_bar.empty()
            status_text.empty()
            return None
            
        except Exception as e:
            st.error(f"Unexpected error during Together AI generation: {str(e)}")
            return None
        """
        Generate image using Replicate API
        
        Args:
            prompt (str): Text prompt for generation
            model (str, optional): Model to use
            parameters (Dict[str, Any], optional): Generation parameters
            
        Returns:
            PIL.Image.Image or None: Generated image
        """
        try:
            # Prepare parameters
            params = parameters or {}
            validated_params = self.validate_parameters(params)
            
            # Create prediction
            payload = {
                "version": model or self.default_model,
                "input": {
                    "prompt": prompt,
                    "width": validated_params['width'],
                    "height": validated_params['height'],
                    "num_inference_steps": validated_params['num_inference_steps'],
                    "guidance_scale": validated_params['guidance_scale']
                }
            }
            
            if validated_params.get('negative_prompt'):
                payload["input"]["negative_prompt"] = validated_params['negative_prompt']
            
            # Start prediction
            response = requests.post(
                self.api_url_base,
                headers=self.headers,
                json=payload,
                timeout=self.settings.api_timeout
            )
            
            if response.status_code != 201:
                st.error(f"Failed to start prediction: {response.text}")
                return None
            
            prediction = response.json()
            prediction_id = prediction['id']
            
            # Poll for completion
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                status_text.text("🎨 Generating image...")
                progress_bar.progress(0.5)
                
                # Check prediction status
                status_response = requests.get(
                    f"{self.api_url_base}/{prediction_id}",
                    headers=self.headers
                )
                
                if status_response.status_code != 200:
                    st.error("Failed to check prediction status")
                    break
                
                status_data = status_response.json()
                
                if status_data['status'] == 'succeeded':
                    # Get image URL
                    image_url = status_data['output'][0]
                    
                    # Download image
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        image = Image.open(io.BytesIO(img_response.content))
                        progress_bar.progress(1.0)
                        status_text.text("✅ Image generated successfully!")
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        return image
                    break
                
                elif status_data['status'] == 'failed':
                    st.error(f"Generation failed: {status_data.get('error', 'Unknown error')}")
                    break
                
                # Wait before next poll
                time.sleep(2)
            
            progress_bar.empty()
            status_text.empty()
            return None
            
        except Exception as e:
            st.error(f"Error with Replicate API: {str(e)}")
            return None

    def generate_image(self, prompt: str, model: Optional[str] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> Optional[Image.Image]:
        """
        Generate image using configured API
        
        Args:
            prompt (str): Text prompt for generation
            model (str, optional): Model to use
            parameters (Dict[str, Any], optional): Generation parameters
            
        Returns:
            PIL.Image.Image or None: Generated image
        """
        # Record start time for performance tracking
        start_time = time.time()
        
        try:
            # Route to appropriate API
            if self.api_type == 'local':
                if hasattr(self, 'local_gen') and self.local_gen:
                    params = parameters or {}
                    validated_params = self.validate_parameters(params)
                    
                    image = self.local_gen.generate_image_local(
                        prompt=prompt,
                        negative_prompt=validated_params.get('negative_prompt', ''),
                        width=validated_params['width'],
                        height=validated_params['height'],
                        num_inference_steps=validated_params['num_inference_steps'],
                        guidance_scale=validated_params['guidance_scale'],
                        seed=validated_params.get('seed')
                    )
                else:
                    st.error("Local generator not available")
                    return None, 0
                    
            elif self.api_type == 'huggingface':
                image = self.generate_image_huggingface(prompt, model, parameters)
            elif self.api_type == 'replicate':
                image = self.generate_image_replicate(prompt, model, parameters)
            elif self.api_type == 'together':
                image = self.generate_image_together(prompt, model, parameters)
            else:
                st.error(f"Unsupported API type: {self.api_type}")
                return None, 0
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            if image:
                st.success(f"🚀 Generation completed in {generation_time:.2f} seconds")
                return image, generation_time
            else:
                st.error("❌ Failed to generate image")
                return None, generation_time
                
        except Exception as e:
            generation_time = time.time() - start_time
            st.error(f"Generation error: {str(e)}")
            return None, generation_time

    def get_available_models(self) -> List[str]:
        """
        Get list of available models for the configured API
        
        Returns:
            List[str]: List of model names
        """
        if self.api_type == 'huggingface':
            return [
                "auto (Recommended - HF picks best model)",
                "CompVis/stable-diffusion-v1-4",
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1",
                "prompthero/openjourney",
                "dreamlike-art/dreamlike-diffusion-1.0",
                "nitrosocke/Arcane-Diffusion",
                "wavymulder/Analog-Diffusion"
            ]
        elif self.api_type == 'replicate':
            return [
                "stability-ai/stable-diffusion",
                "stability-ai/stable-diffusion-xl",
                "ai-forever/kandinsky-2.2"
            ]
        elif self.api_type == 'local':
            return [
                "runwayml/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4",
                "dreamlike-art/dreamlike-diffusion-1.0"
            ]
        else:
            # Fallback - return default model if it exists, otherwise a safe default
            return [getattr(self, 'default_model', 'runwayml/stable-diffusion-v1-5')]

    def test_api_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            bool: True if connection successful
        """
        try:
            if self.api_type == 'huggingface':
                # Try multiple models to find one that works
                test_models = [
                    "CompVis/stable-diffusion-v1-4",
                    "runwayml/stable-diffusion-v1-5", 
                    "prompthero/openjourney",
                    "dreamlike-art/dreamlike-diffusion-1.0",
                    "nitrosocke/Arcane-Diffusion",
                    "wavymulder/Analog-Diffusion"
                ]
                
                st.write("🔍 Testing Hugging Face API with multiple models...")
                
                for i, model in enumerate(test_models):
                    test_url = f"{self.api_url_base}/{model}"
                    try:
                        st.write(f"Testing {i+1}/{len(test_models)}: {model}")
                        
                        response = requests.post(
                            test_url,
                            headers=self.headers,
                            json={"inputs": "test image"},
                            timeout=15
                        )
                        
                        st.write(f"  → Status: {response.status_code}")
                        
                        if response.status_code == 200:
                            st.success(f"✅ SUCCESS! {model} works perfectly!")
                            self.default_model = model
                            return True
                        elif response.status_code == 503:
                            st.info(f"⏳ {model} is loading, might work with retry")
                            continue
                        elif response.status_code == 400:
                            st.warning(f"⚠️ {model} - bad request but auth works")
                            self.default_model = model
                            return True
                        elif response.status_code == 403:
                            st.error(f"❌ {model} - permission denied")
                            continue
                        else:
                            st.warning(f"⚠️ {model} - status {response.status_code}")
                            continue
                            
                    except requests.exceptions.RequestException as e:
                        st.write(f"  → Network error: {str(e)}")
                        continue
                
                st.error("❌ No models worked with your current token")
                st.info("**Try these solutions:**")
                st.info("1. Wait 24-48 hours for new account activation")
                st.info("2. Verify your email address")  
                st.info("3. Create token with 'Write' permissions")
                st.info("4. Try a different Hugging Face account")
                
                return False
            
            return False
            
        except Exception as e:
            st.error(f"API connection test failed: {str(e)}")
            return False

    def estimate_generation_time(self, parameters: Dict[str, Any]) -> float:
        """
        Estimate generation time based on parameters
        
        Args:
            parameters (Dict[str, Any]): Generation parameters
            
        Returns:
            float: Estimated time in seconds
        """
        # Base time estimates (in seconds)
        base_times = {
            'huggingface': 15,
            'replicate': 25,
            'together': 10
        }
        
        base_time = base_times.get(self.api_type, 20)
        
        # Adjust based on parameters
        width = parameters.get('width', 512)
        height = parameters.get('height', 512)
        steps = parameters.get('num_inference_steps', 20)
        
        # Higher resolution increases time
        resolution_factor = (width * height) / (512 * 512)
        time_adjustment = base_time * resolution_factor
        
        # More steps increase time
        steps_factor = steps / 20
        time_adjustment *= steps_factor
        
        return max(5, min(120, time_adjustment))  # Clamp between 5-120 seconds