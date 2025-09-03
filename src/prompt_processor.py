"""
Prompt Processor for Generative LexiArt
Uses LangChain to enhance and optimize prompts for better image generation
"""

from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, List, Optional
import re
import streamlit as st

class PromptEnhancer:
    """
    Enhances user prompts to improve image generation quality
    Uses LangChain for intelligent prompt processing
    """
    
    def __init__(self):
        """Initialize prompt enhancer with templates and rules"""
        
        # Quality enhancement keywords
        self.quality_keywords = [
            "high quality", "detailed", "masterpiece", "professional",
            "sharp focus", "highly detailed", "ultra detailed", "4k", "8k",
            "photorealistic", "hyperrealistic", "cinematic lighting"
        ]
        
        # Style enhancement keywords
        self.style_keywords = {
            "artistic": ["oil painting", "watercolor", "digital art", "concept art"],
            "photographic": ["photography", "portrait", "landscape", "studio lighting"],
            "fantasy": ["fantasy art", "magical", "ethereal", "mystical"],
            "sci-fi": ["futuristic", "cyberpunk", "space", "technological"],
            "vintage": ["retro", "vintage", "classic", "nostalgic"]
        }
        
        # Negative prompt defaults (things to avoid)
        self.default_negative_prompts = [
            "blurry", "low quality", "pixelated", "distorted", "ugly",
            "deformed", "bad anatomy", "extra limbs", "watermark", "text"
        ]
        
        # Common prompt patterns to improve
        self.enhancement_patterns = {
            "simple_object": r"^[a-zA-Z\s]{3,15}$",  # Simple words like "cat", "house"
            "basic_scene": r"^(a|an|the)\s+\w+(\s+\w+){0,3}$",  # "a red car"
            "needs_style": r"^(?!.*(art|style|painting|photo)).+$"  # Missing style keywords
        }

    def detect_prompt_style(self, prompt: str) -> str:
        """
        Detect the intended style of the prompt
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Detected style category
        """
        prompt_lower = prompt.lower()
        
        # Check for style keywords
        for style, keywords in self.style_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return style
        
        # Default style detection based on content
        if any(word in prompt_lower for word in ["person", "portrait", "face", "human"]):
            return "photographic"
        elif any(word in prompt_lower for word in ["dragon", "magic", "castle", "wizard"]):
            return "fantasy"
        elif any(word in prompt_lower for word in ["robot", "future", "cyber", "space"]):
            return "sci-fi"
        else:
            return "artistic"

    def enhance_prompt_quality(self, prompt: str, style: str = "auto") -> str:
        """
        Enhance prompt with quality and style improvements
        
        Args:
            prompt (str): Original prompt
            style (str): Style preference or "auto" for detection
            
        Returns:
            str: Enhanced prompt
        """
        # Detect style if auto
        if style == "auto":
            style = self.detect_prompt_style(prompt)
        
        enhanced_prompt = prompt.strip()
        
        # Add quality keywords if not present
        has_quality = any(keyword in enhanced_prompt.lower() for keyword in self.quality_keywords)
        if not has_quality:
            enhanced_prompt += ", high quality, detailed, masterpiece"
        
        # Add style-specific enhancements
        style_additions = {
            "artistic": ", digital art, concept art, trending on artstation",
            "photographic": ", professional photography, sharp focus, bokeh",
            "fantasy": ", fantasy art, magical atmosphere, ethereal lighting",
            "sci-fi": ", futuristic, high-tech, cinematic lighting",
            "vintage": ", vintage style, nostalgic, film grain"
        }
        
        if style in style_additions:
            enhanced_prompt += style_additions[style]
        
        return enhanced_prompt

    def generate_negative_prompt(self, positive_prompt: str) -> str:
        """
        Generate appropriate negative prompt based on positive prompt
        
        Args:
            positive_prompt (str): The main generation prompt
            
        Returns:
            str: Generated negative prompt
        """
        negative_elements = self.default_negative_prompts.copy()
        
        # Add specific negative prompts based on content
        prompt_lower = positive_prompt.lower()
        
        if "person" in prompt_lower or "portrait" in prompt_lower:
            negative_elements.extend([
                "multiple faces", "extra heads", "missing eyes",
                "deformed hands", "extra fingers"
            ])
        
        if "landscape" in prompt_lower or "nature" in prompt_lower:
            negative_elements.extend([
                "buildings", "urban", "people", "vehicles"
            ])
        
        if "animal" in prompt_lower:
            negative_elements.extend([
                "extra legs", "deformed animal", "human features"
            ])
        
        return ", ".join(negative_elements)

    def extract_parameters_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Extract generation parameters mentioned in the prompt
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        parameters = {}
        
        # Extract resolution mentions
        resolution_patterns = [
            r"(\d+)x(\d+)",  # 512x512
            r"(\d+k)",       # 4k, 8k
        ]
        
        for pattern in resolution_patterns:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                if 'k' in matches[0]:
                    # Convert 4k, 8k to actual dimensions
                    k_value = int(matches[0].replace('k', ''))
                    if k_value == 4:
                        parameters['width'] = parameters['height'] = 768
                    elif k_value == 8:
                        parameters['width'] = parameters['height'] = 1024
                else:
                    # Direct width x height
                    parameters['width'] = int(matches[0][0])
                    parameters['height'] = int(matches[0][1])
                break
        
        # Extract style strength mentions
        if "subtle" in prompt.lower():
            parameters['guidance_scale'] = 5.0
        elif "strong" in prompt.lower() or "bold" in prompt.lower():
            parameters['guidance_scale'] = 12.0
        
        return parameters

    def clean_prompt(self, prompt: str) -> str:
        """
        Clean prompt by removing parameter mentions and normalizing
        
        Args:
            prompt (str): Raw prompt
            
        Returns:
            str: Cleaned prompt
        """
        # Remove resolution mentions
        cleaned = re.sub(r'\b\d+x\d+\b', '', prompt)
        cleaned = re.sub(r'\b\d+k\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove strength mentions
        strength_words = ['subtle', 'strong', 'bold', 'weak', 'intense']
        for word in strength_words:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r',\s*,', ',', cleaned)  # Double commas
        cleaned = cleaned.strip(' ,')  # Leading/trailing spaces and commas
        
        return cleaned

    def suggest_improvements(self, prompt: str) -> List[str]:
        """
        Suggest improvements for the given prompt
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        prompt_lower = prompt.lower()
        
        # Check prompt length
        if len(prompt.split()) < 3:
            suggestions.append("Consider adding more descriptive details to your prompt")
        
        # Check for quality keywords
        if not any(keyword in prompt_lower for keyword in self.quality_keywords):
            suggestions.append("Add quality keywords like 'high quality', 'detailed', or 'masterpiece'")
        
        # Check for style specification
        has_style = any(
            any(keyword in prompt_lower for keyword in keywords)
            for keywords in self.style_keywords.values()
        )
        if not has_style:
            suggestions.append("Specify an art style (e.g., 'digital art', 'photography', 'oil painting')")
        
        # Check for lighting specification
        lighting_keywords = ["lighting", "lit", "bright", "dark", "shadow", "sunlight"]
        if not any(keyword in prompt_lower for keyword in lighting_keywords):
            suggestions.append("Consider adding lighting description (e.g., 'soft lighting', 'dramatic shadows')")
        
        # Check for color specification
        color_keywords = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "colorful", "vibrant"]
        if not any(keyword in prompt_lower for keyword in color_keywords):
            suggestions.append("Consider specifying colors or color scheme")
        
        return suggestions

class PromptProcessor:
    """
    Main prompt processing class that coordinates all prompt enhancement features
    """
    
    def __init__(self):
        """Initialize the prompt processor"""
        self.enhancer = PromptEnhancer()
        self.prompt_history = []  # Store recent prompts for suggestions

    def process_prompt(self, user_prompt: str, enhancement_level: str = "medium",
                      custom_style: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user prompt with specified enhancement level
        
        Args:
            user_prompt (str): Original user input
            enhancement_level (str): Level of enhancement ("light", "medium", "heavy")
            custom_style (str, optional): Custom style preference
            
        Returns:
            Dict[str, Any]: Processed prompt data
        """
        # Clean the input prompt
        cleaned_prompt = self.enhancer.clean_prompt(user_prompt)
        
        # Extract parameters from original prompt
        extracted_params = self.enhancer.extract_parameters_from_prompt(user_prompt)
        
        # Enhance based on level
        if enhancement_level == "light":
            enhanced_prompt = cleaned_prompt
        elif enhancement_level == "medium":
            style = custom_style or "auto"
            enhanced_prompt = self.enhancer.enhance_prompt_quality(cleaned_prompt, style)
        else:  # heavy
            style = custom_style or "auto"
            enhanced_prompt = self.enhancer.enhance_prompt_quality(cleaned_prompt, style)
            # Add more aggressive enhancements for heavy level
            enhanced_prompt += ", ultra detailed, 8k resolution, award winning"
        
        # Generate negative prompt
        negative_prompt = self.enhancer.generate_negative_prompt(enhanced_prompt)
        
        # Get improvement suggestions
        suggestions = self.enhancer.suggest_improvements(user_prompt)
        
        # Add to history
        self.prompt_history.append(user_prompt)
        if len(self.prompt_history) > 50:  # Keep only last 50
            self.prompt_history.pop(0)
        
        return {
            'original_prompt': user_prompt,
            'cleaned_prompt': cleaned_prompt,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt': negative_prompt,
            'extracted_parameters': extracted_params,
            'detected_style': self.enhancer.detect_prompt_style(cleaned_prompt),
            'suggestions': suggestions,
            'enhancement_level': enhancement_level
        }

    def get_prompt_templates(self) -> Dict[str, str]:
        """
        Get pre-defined prompt templates for different categories
        
        Returns:
            Dict[str, str]: Template categories and their prompts
        """
        templates = {
            "Portrait": "professional portrait of {subject}, studio lighting, high quality, detailed face, sharp focus",
            "Landscape": "beautiful landscape of {location}, golden hour lighting, panoramic view, high resolution, nature photography",
            "Fantasy": "fantasy art of {subject}, magical atmosphere, ethereal lighting, detailed, concept art, trending on artstation",
            "Sci-Fi": "futuristic {subject}, cyberpunk style, neon lighting, high-tech, cinematic, digital art",
            "Abstract": "abstract art representing {concept}, colorful, modern art style, creative composition",
            "Vintage": "vintage style {subject}, retro aesthetic, film grain, nostalgic atmosphere, classic photography",
            "Minimalist": "minimalist {subject}, clean composition, simple, elegant, white background, professional"
        }
        return templates

    def apply_template(self, template: str, subject: str) -> str:
        """
        Apply a template with the given subject
        
        Args:
            template (str): Template string with {subject} placeholder
            subject (str): Subject to insert into template
            
        Returns:
            str: Formatted prompt
        """
        return template.format(subject=subject)

    def get_similar_prompts(self, current_prompt: str, limit: int = 5) -> List[str]:
        """
        Get similar prompts from history
        
        Args:
            current_prompt (str): Current prompt to find similar ones
            limit (int): Maximum number of similar prompts
            
        Returns:
            List[str]: Similar prompts from history
        """
        if not self.prompt_history:
            return []
        
        current_words = set(current_prompt.lower().split())
        similar_prompts = []
        
        for historical_prompt in self.prompt_history:
            if historical_prompt == current_prompt:
                continue
                
            historical_words = set(historical_prompt.lower().split())
            common_words = current_words.intersection(historical_words)
            
            if len(common_words) >= 2:  # At least 2 common words
                similarity = len(common_words) / max(len(current_words), len(historical_words))
                similar_prompts.append((historical_prompt, similarity))
        
        # Sort by similarity and return top results
        similar_prompts.sort(key=lambda x: x[1], reverse=True)
        return [prompt for prompt, _ in similar_prompts[:limit]]

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate prompt and return validation results
        
        Args:
            prompt (str): Prompt to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'word_count': len(prompt.split()),
            'character_count': len(prompt)
        }
        
        # Check prompt length
        if len(prompt.strip()) < 3:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Prompt is too short")
        
        if len(prompt) > 1000:
            validation_result['warnings'].append("Prompt is very long and may be truncated")
        
        # Check for potentially problematic content
        problematic_words = ['nsfw', 'explicit', 'gore', 'violence']
        if any(word in prompt.lower() for word in problematic_words):
            validation_result['warnings'].append("Prompt may contain inappropriate content")
        
        # Check for excessive repetition
        words = prompt.lower().split()
        unique_words = set(words)
        if len(words) > 10 and len(unique_words) / len(words) < 0.5:
            validation_result['warnings'].append("Prompt contains excessive word repetition")
        
        return validation_result