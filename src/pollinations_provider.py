import requests
import time
from typing import Optional, Dict, Any
from PIL import Image
import io
import streamlit as st
import urllib.parse
from urllib.parse import quote

class PollinationsProvider:
    """
    FIXED Pollinations provider that enforces 1024x1024 generation
    """
    
    def __init__(self):
        """Initialize with fixed 1024x1024 settings"""
        self.api_url = "https://image.pollinations.ai/prompt"
        self.max_retries = 3
        self.base_timeout = 45
        self.retry_delay = 2
        
        # FIXED: Force 1024x1024 always
        self.forced_width = 1024
        self.forced_height = 1024
        
    def generate_image(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Image.Image]:
        """
        Generate image with FORCED 1024x1024 resolution
        """
        # Clean and validate prompt
        cleaned_prompt = self._clean_prompt_for_url(prompt)
        if not cleaned_prompt or len(cleaned_prompt.strip()) < 3:
            st.error("Prompt is too short or invalid")
            return None
        
        # IGNORE input parameters, always use 1024x1024
        forced_params = {
            'width': self.forced_width,
            'height': self.forced_height
        }
        
        # Show progress
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Try generation with multiple approaches
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    # First attempt: Direct 1024x1024 generation
                    progress_placeholder.info(f"🎨 Generating 1024×1024 image... (Attempt {attempt + 1})")
                    image = self._generate_forced_1024(cleaned_prompt, status_placeholder)
                    
                elif attempt == 1:
                    # Second attempt: Different API parameters
                    progress_placeholder.info(f"🔧 Trying alternative approach... (Attempt {attempt + 1})")
                    image = self._generate_alternative_1024(cleaned_prompt, status_placeholder)
                    
                else:
                    # Final attempt: Basic generation then resize
                    progress_placeholder.info(f"⚡ Fallback generation... (Final attempt)")
                    image = self._generate_and_resize_1024(cleaned_prompt, status_placeholder)
                
                if image:
                    # CRITICAL: Verify the image is actually 1024x1024
                    if image.size != (1024, 1024):
                        st.warning(f"Service returned {image.size[0]}×{image.size[1]}, resizing to 1024×1024...")
                        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                    
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    st.success(f"✅ Generated 1024×1024 image successfully")
                    return image
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    status_placeholder.warning(f"⏱️ Timeout (attempt {attempt + 1}), retrying...")
                    time.sleep(self.retry_delay)
                else:
                    st.error("⏱️ Generation timed out. Please try again.")
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    status_placeholder.warning(f"⚠️ Error: {str(e)}, retrying...")
                    time.sleep(self.retry_delay)
                else:
                    st.error(f"❌ Generation failed: {str(e)}")
        
        # All attempts failed
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error("Failed to generate 1024×1024 image after multiple attempts")
        return None
    
    def _generate_forced_1024(self, prompt: str, status_placeholder) -> Optional[Image.Image]:
        """Primary method: Force 1024x1024 with explicit parameters"""
        try:
            # Construct URL with FORCED 1024x1024
            image_url = f"{self.api_url}/{prompt}"
            
            url_params = {
                'width': 1024,
                'height': 1024,
                'model': 'flux',  # Use best model
                'enhance': 'true',
                'nologo': 'true',
                'nofeed': 'true'
            }
            
            status_placeholder.info("🚀 Requesting 1024×1024 generation...")
            
            response = requests.get(
                image_url,
                params=url_params,
                timeout=self.base_timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/png,image/jpeg,image/*;q=0.9',
                    'Cache-Control': 'no-cache'
                },
                stream=True
            )
            
            return self._process_response(response)
            
        except Exception as e:
            raise e
    
    def _generate_alternative_1024(self, prompt: str, status_placeholder) -> Optional[Image.Image]:
        """Alternative method with different parameters"""
        try:
            # Try different parameter format
            image_url = f"{self.api_url}/{prompt}?width=1024&height=1024&model=turbo&enhance=true"
            
            status_placeholder.info("🔄 Using alternative 1024×1024 method...")
            
            response = requests.get(
                image_url,
                timeout=self.base_timeout + 15,
                headers={'User-Agent': 'Mozilla/5.0'},
                stream=True
            )
            
            return self._process_response(response)
            
        except Exception as e:
            raise e
    
    def _generate_and_resize_1024(self, prompt: str, status_placeholder) -> Optional[Image.Image]:
        """Fallback: Generate at any size then resize to 1024x1024"""
        try:
            # Simple generation without size constraints
            image_url = f"https://image.pollinations.ai/prompt/{prompt}"
            
            status_placeholder.info("📐 Generating and resizing to 1024×1024...")
            
            response = requests.get(
                image_url,
                timeout=60,
                headers={'User-Agent': 'Mozilla/5.0'},
                stream=True
            )
            
            image = self._process_response(response)
            
            if image:
                # Force resize to 1024x1024
                resized_image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                status_placeholder.info("✅ Resized to 1024×1024")
                return resized_image
            
            return None
            
        except Exception as e:
            raise e
    
    def _process_response(self, response: requests.Response) -> Optional[Image.Image]:
        """Process API response and validate image"""
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type.lower():
                return None
            
            image_data = response.content
            if len(image_data) < 1000:  # Too small
                return None
            
            try:
                image = Image.open(io.BytesIO(image_data))
                
                # Validate minimum size
                if image.size[0] < 100 or image.size[1] < 100:
                    return None
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                return image
                
            except Exception:
                return None
        else:
            # Handle error responses
            if response.status_code == 429:
                st.warning("🚫 Rate limit - service is busy")
            elif response.status_code >= 500:
                st.warning("🔧 Service temporarily unavailable")
            return None
    
    def _clean_prompt_for_url(self, prompt: str) -> str:
        """Clean prompt for URL compatibility"""
        if not prompt or len(prompt.strip()) < 3:
            return ""
        
        cleaned = prompt.strip()
        
        # Remove problematic characters
        replacements = {
            '"': '', "'": '', '\n': ' ', '\r': ' ', '\t': ' ',
            '&': 'and', '#': '', '%': '', '+': ' plus ',
            '=': ' equals ', '?': '', '[': '(', ']': ')',
            '{': '(', '}': ')', '|': '', '\\': '',
            '<': '', '>': '', '^': '', '`': '', '~': ''
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Normalize spaces
        cleaned = ' '.join(cleaned.split())
        
        # Truncate if too long
        if len(cleaned) > 200:
            words = cleaned.split()
            while len(' '.join(words)) > 200 and words:
                words.pop()
            cleaned = ' '.join(words)
        
        # URL encode
        try:
            cleaned = quote(cleaned, safe=' ')
            return cleaned
        except Exception:
            return cleaned.replace(' ', '%20')
    
    def is_available(self) -> bool:
        """Check if service is available"""
        try:
            response = requests.head(
                "https://image.pollinations.ai", 
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            return response.status_code in [200, 404]
        except Exception:
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        status = {
            'available': False,
            'response_time': None,
            'resolution': '1024×1024',
            'last_check': time.time()
        }
        
        try:
            start_time = time.time()
            response = requests.head(
                "https://image.pollinations.ai",
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response_time = time.time() - start_time
            
            status['available'] = response.status_code in [200, 404]
            status['response_time'] = round(response_time, 2)
            
        except Exception as e:
            status['error'] = str(e)
        
        return status
