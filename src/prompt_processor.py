from typing import Dict, Any, List, Optional
import re
import streamlit as st

class PromptProcessor:
    """
    Complete prompt processing class with integrated enhancement capabilities
    """
    
    def __init__(self, enable_llm: bool = True):
        """Initialize the comprehensive prompt processor with all enhancement features"""
        self.prompt_history = []
        self.enable_llm = enable_llm
        self.llm_enhancer = None
        
        # Quality enhancement keywords
        self.quality_keywords = [
            "high quality", "detailed", "masterpiece", "professional",
            "sharp focus", "highly detailed", "ultra detailed", "4k", "8k",
            "photorealistic", "hyperrealistic", "cinematic lighting",
            "award winning", "trending on artstation", "fine art"
        ]
        
        # Comprehensive style enhancement keywords
        self.style_keywords = {
            "artistic": [
                "oil painting", "watercolor", "digital art", "concept art", 
                "acrylic painting", "fine art", "artistic", "painterly",
                "brushstrokes", "canvas", "gallery quality"
            ],
            "photographic": [
                "photography", "portrait", "landscape", "studio lighting",
                "professional photography", "DSLR", "bokeh", "depth of field",
                "natural lighting", "golden hour", "cinematic photography"
            ],
            "fantasy": [
                "fantasy art", "magical", "ethereal", "mystical", "enchanted",
                "otherworldly", "epic fantasy", "medieval", "dragons",
                "wizards", "fairy tale", "mythological"
            ],
            "sci-fi": [
                "futuristic", "cyberpunk", "space", "technological", "sci-fi",
                "neon lights", "holographic", "android", "alien", "spaceship",
                "dystopian", "utopian", "robotic"
            ],
            "vintage": [
                "retro", "vintage", "classic", "nostalgic", "antique",
                "old-fashioned", "sepia", "film grain", "aged", "historical",
                "period piece", "timeless"
            ],
            "anime": [
                "anime", "manga", "japanese animation", "cel shading",
                "anime style", "kawaii", "chibi", "studio ghibli style",
                "anime character", "otaku"
            ],
            "realistic": [
                "realistic", "lifelike", "true to life", "natural",
                "authentic", "genuine", "real world", "documentary style"
            ]
        }
        
        # Lighting keywords for enhancement
        self.lighting_keywords = [
            "soft lighting", "dramatic lighting", "natural lighting",
            "studio lighting", "golden hour", "blue hour", "sunset lighting",
            "rim lighting", "backlighting", "side lighting", "ambient lighting",
            "volumetric lighting", "god rays", "chiaroscuro"
        ]
        
        # Composition keywords
        self.composition_keywords = [
            "rule of thirds", "centered composition", "symmetrical",
            "dynamic angle", "bird's eye view", "worm's eye view",
            "close-up", "wide shot", "medium shot", "macro photography",
            "panoramic", "portrait orientation", "landscape orientation"
        ]
        
        # Negative prompt defaults
        self.default_negative_prompts = [
            "blurry", "low quality", "pixelated", "distorted", "ugly",
            "deformed", "bad anatomy", "extra limbs", "watermark", "text",
            "signature", "username", "error", "cropped", "worst quality",
            "jpeg artifacts", "duplicate", "morbid", "mutilated"
        ]
        
        # Style-specific negative prompts
        self.style_negative_prompts = {
            "photographic": ["cartoon", "anime", "painting", "drawing", "sketch"],
            "artistic": ["photograph", "photo", "realistic", "camera"],
            "fantasy": ["modern", "contemporary", "urban", "realistic"],
            "sci-fi": ["medieval", "ancient", "primitive", "natural"],
            "vintage": ["modern", "futuristic", "contemporary", "digital"]
        }
        

    def detect_prompt_style(self, prompt: str) -> str:
        """Advanced style detection with scoring system"""
        prompt_lower = prompt.lower()
        style_scores = {}
        
        # Score each style based on keyword matches
        for style, keywords in self.style_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
            style_scores[style] = score
        
        # Find the highest scoring style
        best_style = max(style_scores, key=style_scores.get)
        
        # If no clear winner, use content-based detection
        if style_scores[best_style] == 0:
            if any(word in prompt_lower for word in ["person", "portrait", "face", "human", "people"]):
                return "photographic"
            elif any(word in prompt_lower for word in ["dragon", "magic", "castle", "wizard", "fantasy"]):
                return "fantasy"
            elif any(word in prompt_lower for word in ["robot", "future", "cyber", "space", "alien"]):
                return "sci-fi"
            elif any(word in prompt_lower for word in ["anime", "manga", "kawaii", "chibi"]):
                return "anime"
            elif any(word in prompt_lower for word in ["old", "vintage", "retro", "classic"]):
                return "vintage"
            else:
                return "artistic"
        
        return best_style

    def enhance_prompt_quality(self, prompt: str, style: str = "auto", enhancement_level: str = "medium") -> str:
        """Comprehensive prompt enhancement with multiple levels"""
        if style == "auto":
            style = self.detect_prompt_style(prompt)
        
        enhanced_prompt = prompt.strip()
        
        # Enhancement based on level
        if enhancement_level == "light":
            # Just add basic quality if missing
            if not any(keyword in enhanced_prompt.lower() for keyword in ["quality", "detailed"]):
                enhanced_prompt += ", high quality"
        
        elif enhancement_level == "medium":
            # Add quality keywords if not present
            has_quality = any(keyword in enhanced_prompt.lower() for keyword in self.quality_keywords[:5])
            if not has_quality:
                enhanced_prompt += ", high quality, detailed, professional"
            
            # Add lighting if not mentioned
            has_lighting = any(keyword in enhanced_prompt.lower() for keyword in self.lighting_keywords)
            if not has_lighting:
                enhanced_prompt += ", beautiful lighting"
        
        elif enhancement_level == "heavy":
            # Comprehensive enhancement
            has_quality = any(keyword in enhanced_prompt.lower() for keyword in self.quality_keywords)
            if not has_quality:
                enhanced_prompt += ", high quality, detailed, masterpiece, professional"
            
            has_lighting = any(keyword in enhanced_prompt.lower() for keyword in self.lighting_keywords)
            if not has_lighting:
                enhanced_prompt += ", cinematic lighting, dramatic lighting"
            
            # Add composition keywords
            has_composition = any(keyword in enhanced_prompt.lower() for keyword in self.composition_keywords)
            if not has_composition:
                enhanced_prompt += ", perfect composition"
            
            # Add final quality boost
            enhanced_prompt += ", award winning, trending on artstation"
        
        # Add style-specific enhancements
        style_additions = {
            "artistic": ", digital art, concept art, painterly, artistic masterpiece",
            "photographic": ", professional photography, sharp focus, bokeh, DSLR quality",
            "fantasy": ", fantasy art, magical atmosphere, ethereal lighting, epic fantasy",
            "sci-fi": ", futuristic, high-tech, cyberpunk aesthetic, sci-fi concept art",
            "vintage": ", vintage style, retro aesthetic, film grain, nostalgic atmosphere",
            "anime": ", anime style, manga art, cel shading, studio quality animation",
            "realistic": ", photorealistic, lifelike, true to life, realistic rendering"
        }
        
        if style in style_additions:
            style_enhancement = style_additions[style]
            # Only add if not already present
            if not any(word in enhanced_prompt.lower() for word in style_enhancement.split(", ")):
                enhanced_prompt += style_enhancement
        
        return enhanced_prompt

    def generate_negative_prompt(self, positive_prompt: str, detected_style: str = "artistic") -> str:
        """Generate comprehensive negative prompt based on positive prompt and style"""
        negative_elements = self.default_negative_prompts.copy()
        prompt_lower = positive_prompt.lower()
        
        # Add content-specific negative prompts
        if "person" in prompt_lower or "portrait" in prompt_lower or "human" in prompt_lower:
            negative_elements.extend([
                "multiple faces", "extra heads", "missing eyes", "deformed hands", 
                "extra fingers", "missing fingers", "extra arms", "extra legs",
                "malformed hands", "poorly drawn hands", "poorly drawn face"
            ])
        
        if "landscape" in prompt_lower or "nature" in prompt_lower:
            negative_elements.extend([
                "people", "humans", "buildings", "urban", "city"
            ])
        
        if "animal" in prompt_lower:
            negative_elements.extend([
                "extra legs", "deformed animal", "human features on animals",
                "multiple tails", "extra eyes"
            ])
        
        # Add style-specific negative prompts
        if detected_style in self.style_negative_prompts:
            negative_elements.extend(self.style_negative_prompts[detected_style])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_negatives = []
        for item in negative_elements:
            if item not in seen:
                seen.add(item)
                unique_negatives.append(item)
        
        return ", ".join(unique_negatives)

    def extract_parameters_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Extract generation parameters mentioned in the prompt"""
        parameters = {}
        prompt_lower = prompt.lower()
        
        # Extract resolution mentions
        resolution_patterns = [
            r"(\d+)x(\d+)",  # 512x512
            r"(\d+k)",       # 4k, 8k
            r"high resolution", r"low resolution"
        ]
        
        for pattern in resolution_patterns:
            matches = re.findall(pattern, prompt_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    # Direct width x height
                    parameters['width'] = int(matches[0][0])
                    parameters['height'] = int(matches[0][1])
                elif 'k' in str(matches[0]):
                    # Convert 4k, 8k to actual dimensions
                    k_value = int(str(matches[0]).replace('k', ''))
                    if k_value == 4:
                        parameters['width'] = parameters['height'] = 1024
                    elif k_value == 8:
                        parameters['width'] = parameters['height'] = 1024  # Keep at 1024 for our app
                break
        
        # Extract quality/detail level mentions
        if any(word in prompt_lower for word in ["highly detailed", "ultra detailed", "extremely detailed"]):
            parameters['num_inference_steps'] = 35
        elif any(word in prompt_lower for word in ["detailed", "high quality"]):
            parameters['num_inference_steps'] = 25
        elif any(word in prompt_lower for word in ["simple", "minimalist", "clean"]):
            parameters['num_inference_steps'] = 20
        
        # Extract style strength mentions
        if any(word in prompt_lower for word in ["subtle", "soft", "gentle"]):
            parameters['guidance_scale'] = 6.0
        elif any(word in prompt_lower for word in ["strong", "bold", "dramatic", "intense"]):
            parameters['guidance_scale'] = 12.0
        elif any(word in prompt_lower for word in ["very strong", "extremely", "maximum"]):
            parameters['guidance_scale'] = 15.0
        
        return parameters

    def clean_prompt(self, prompt: str) -> str:
        """Clean prompt by removing parameter mentions and normalizing"""
        # Remove resolution mentions
        cleaned = re.sub(r'\b\d+x\d+\b', '', prompt)
        cleaned = re.sub(r'\b\d+k\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove strength mentions that are parameter-related
        parameter_words = [
            'subtle', 'strong', 'bold', 'weak', 'intense', 'maximum',
            'high resolution', 'low resolution', 'extremely detailed'
        ]
        for word in parameter_words:
            cleaned = re.sub(rf'\b{re.escape(word)}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r',\s*,', ',', cleaned)  # Double commas
        cleaned = re.sub(r'\s*,\s*$', '', cleaned)  # Trailing comma
        cleaned = cleaned.strip(' ,')  # Leading/trailing spaces and commas
        
        return cleaned

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Comprehensive prompt improvement suggestions"""
        suggestions = []
        prompt_lower = prompt.lower()
        
        # Check prompt length
        word_count = len(prompt.split())
        if word_count < 5:
            suggestions.append("Add more descriptive details - try to use 10-20 words for better results")
        elif word_count > 50:
            suggestions.append("Consider shortening your prompt - very long prompts can be less effective")
        
        # Check for quality keywords
        if not any(keyword in prompt_lower for keyword in self.quality_keywords):
            suggestions.append("Add quality keywords like 'high quality', 'detailed', or 'masterpiece'")
        
        # Check for style specification
        has_style = any(
            any(keyword in prompt_lower for keyword in keywords)
            for keywords in self.style_keywords.values()
        )
        if not has_style:
            suggestions.append("Specify an art style (e.g., 'digital art', 'photography', 'fantasy art')")
        
        # Check for lighting specification
        if not any(keyword in prompt_lower for keyword in self.lighting_keywords):
            suggestions.append("Add lighting description (e.g., 'soft lighting', 'dramatic lighting', 'golden hour')")
        
        # Check for color specification
        color_keywords = [
            "red", "blue", "green", "yellow", "purple", "orange", "black", "white",
            "colorful", "vibrant", "muted", "pastel", "bright", "dark", "monochrome"
        ]
        if not any(keyword in prompt_lower for keyword in color_keywords):
            suggestions.append("Consider specifying colors or color scheme")
        
        # Check for composition/framing
        if not any(keyword in prompt_lower for keyword in self.composition_keywords):
            suggestions.append("Add composition details (e.g., 'close-up', 'wide shot', 'centered')")
        
        # Check for mood/atmosphere
        mood_keywords = [
            "peaceful", "dramatic", "mysterious", "cheerful", "dark", "bright",
            "moody", "atmospheric", "serene", "energetic", "calm", "intense"
        ]
        if not any(keyword in prompt_lower for keyword in mood_keywords):
            suggestions.append("Consider adding mood or atmosphere (e.g., 'peaceful', 'dramatic', 'mysterious')")
        
        return suggestions

    def calculate_quality_score(self, prompt: str) -> float:
        """Calculate a quality score for the prompt (0-100)"""
        score = 0
        prompt_lower = prompt.lower()
        
        # Word count (optimal range: 10-30 words)
        word_count = len(prompt.split())
        if 10 <= word_count <= 30:
            score += 25
        elif 5 <= word_count < 10 or 30 < word_count <= 50:
            score += 15
        
        # Quality keywords present
        quality_words = sum(1 for keyword in self.quality_keywords if keyword in prompt_lower)
        score += min(quality_words * 5, 25)
        
        # Style keywords present
        style_words = sum(1 for keywords in self.style_keywords.values() 
                         for keyword in keywords if keyword in prompt_lower)
        score += min(style_words * 3, 20)
        
        # Lighting keywords
        lighting_words = sum(1 for keyword in self.lighting_keywords if keyword in prompt_lower)
        score += min(lighting_words * 5, 15)
        
        # Composition keywords
        comp_words = sum(1 for keyword in self.composition_keywords if keyword in prompt_lower)
        score += min(comp_words * 3, 15)
        
        return min(score, 100)

    def process_prompt(self, user_prompt: str, enhancement_level: str = "medium",
                      custom_style: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
        """Comprehensive prompt processing with full enhancement pipeline"""
        # Step 1: Clean the input prompt
        cleaned_prompt = self.clean_prompt(user_prompt)
        
        # Step 2: Extract parameters from prompt
        extracted_params = self.extract_parameters_from_prompt(user_prompt)
        
        # Step 3: Detect style
        detected_style = self.detect_prompt_style(cleaned_prompt)
        final_style = custom_style or detected_style
        
        # Step 4: LLM Enhancement (if available and requested)
        llm_results = None
        if enhancement_level == "llm" and self.llm_enhancer and use_llm:
            try:
                st.info("Using advanced AI enhancement...")
                llm_results = self.llm_enhancer.smart_prompt_enhance(cleaned_prompt)
                
                if llm_results and llm_results.get("success"):
                    enhanced_prompt = llm_results["enhanced_prompt"]
                    st.success("AI enhancement complete!")
                else:
                    enhanced_prompt = self.enhance_prompt_quality(
                        cleaned_prompt, final_style, enhancement_level
                    )
                    st.warning("AI enhancement failed, using advanced rule-based enhancement")
            except Exception as e:
                enhanced_prompt = self.enhance_prompt_quality(
                    cleaned_prompt, final_style, enhancement_level
                )
                st.warning(f"AI enhancement error: {str(e)}")
        
        # Step 5: Rule-based enhancement
        else:
            enhanced_prompt = self.enhance_prompt_quality(
                cleaned_prompt, final_style, enhancement_level
            )
        
        # Step 6: Generate negative prompt
        negative_prompt = self.generate_negative_prompt(enhanced_prompt, detected_style)
        
        # Step 7: Get improvement suggestions
        suggestions = self.suggest_improvements(user_prompt)
        
        # Step 8: Add to history
        self.prompt_history.append(user_prompt)
        if len(self.prompt_history) > 100:  # Keep more history
            self.prompt_history.pop(0)
        
        # Step 9: Compile comprehensive results
        result = {
            'original_prompt': user_prompt,
            'cleaned_prompt': cleaned_prompt,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt': negative_prompt,
            'extracted_parameters': extracted_params,
            'detected_style': detected_style,
            'final_style': final_style,
            'suggestions': suggestions,
            'enhancement_level': enhancement_level,
            'llm_enhanced': enhancement_level == "llm" and llm_results and llm_results.get("success", False),
            'word_count': len(enhanced_prompt.split()),
            'quality_score': self.calculate_quality_score(enhanced_prompt)
        }
        
        # Add LLM results if available
        if llm_results and llm_results.get("success"):
            result.update({
                'llm_results': llm_results,
                'style_recommendations': llm_results.get('style_recommendations', []),
                'creative_variations': llm_results.get('creative_variations', []),
                'prompt_analysis': llm_results.get('analysis', '')
            })
        
        return result

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Comprehensive prompt validation"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'word_count': len(prompt.split()),
            'character_count': len(prompt),
            'quality_score': self.calculate_quality_score(prompt)
        }
        
        # Check prompt length
        if len(prompt.strip()) < 3:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Prompt is too short - needs at least 3 characters")
        
        if len(prompt) > 1000:
            validation_result['warnings'].append("Prompt is very long and may be truncated by the AI")
        
        # Check for potentially problematic content
        problematic_words = ['nsfw', 'explicit', 'gore', 'violence', 'nude', 'naked']
        if any(word in prompt.lower() for word in problematic_words):
            validation_result['warnings'].append("Prompt may contain inappropriate content")
        
        # Check for excessive repetition
        words = prompt.lower().split()
        unique_words = set(words)
        if len(words) > 10 and len(unique_words) / len(words) < 0.6:
            validation_result['warnings'].append("Prompt contains excessive word repetition")
        
        # Check for conflicting styles
        detected_styles = []
        for style, keywords in self.style_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                detected_styles.append(style)
        
        if len(detected_styles) > 2:
            validation_result['warnings'].append("Multiple conflicting art styles detected")
        
        return validation_result

    def get_prompt_templates(self) -> Dict[str, str]:
        """Get comprehensive prompt templates"""
        templates = {
            "Portrait": "professional portrait of {subject}, studio lighting, high quality, detailed face, sharp focus, photorealistic, beautiful lighting",
            "Landscape": "beautiful landscape of {location}, golden hour lighting, panoramic view, high resolution, nature photography, scenic vista, perfect composition",
            "Fantasy": "fantasy artwork of {subject}, magical atmosphere, ethereal lighting, detailed, concept art, trending on artstation, epic fantasy, mystical",
            "Sci-Fi": "futuristic {subject}, cyberpunk style, neon lighting, high-tech, cinematic, digital art, sci-fi concept art, advanced technology",
            "Abstract": "abstract art representing {concept}, colorful, modern art style, creative composition, artistic interpretation, contemporary",
            "Vintage": "vintage style {subject}, retro aesthetic, film grain, nostalgic atmosphere, classic photography, aged, timeless",
            "Anime": "anime style {subject}, manga art, cel shading, studio quality animation, kawaii, detailed anime character, japanese animation",
            "Digital Art": "digital artwork of {subject}, concept art, detailed illustration, digital painting, artstation quality, modern digital art"
        }
        return templates

    def get_similar_prompts(self, current_prompt: str, limit: int = 5) -> List[str]:
        """Get similar prompts from history with advanced matching"""
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
                # Calculate similarity score
                union_words = current_words.union(historical_words)
                similarity = len(common_words) / len(union_words)
                similar_prompts.append((historical_prompt, similarity))
        
        # Sort by similarity and return top results
        similar_prompts.sort(key=lambda x: x[1], reverse=True)
        return [prompt for prompt, _ in similar_prompts[:limit]]

    def get_enhancement_preview(self, prompt: str, level: str = "medium") -> str:
        """Get a preview of how the prompt would be enhanced"""
        cleaned = self.clean_prompt(prompt)
        style = self.detect_prompt_style(cleaned)
        enhanced = self.enhance_prompt_quality(cleaned, style, level)
        return enhanced

    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt complexity and provide insights"""
        words = prompt.split()
        
        analysis = {
            'word_count': len(words),
            'character_count': len(prompt),
            'unique_words': len(set(word.lower() for word in words)),
            'complexity_score': 0,
            'readability': 'simple'
        }
        
        # Calculate complexity score
        complexity = 0
        
        # Word count factor
        if len(words) > 20:
            complexity += 2
        elif len(words) > 10:
            complexity += 1
        
        # Unique word ratio
        if analysis['unique_words'] / max(len(words), 1) > 0.8:
            complexity += 1
        
        # Style keyword count
        style_count = sum(
            1 for keywords in self.style_keywords.values()
            for keyword in keywords if keyword in prompt.lower()
        )
        complexity += min(style_count, 3)
        
        analysis['complexity_score'] = complexity
        
        if complexity >= 5:
            analysis['readability'] = 'complex'
        elif complexity >= 3:
            analysis['readability'] = 'moderate'
        
        return analysis
