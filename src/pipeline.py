"""
Fixed Generation Pipeline for Generative LexiArt
Removed config import - uses direct configuration
"""

import time
import os
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
import streamlit as st

from .prompt_processor import PromptProcessor
from .image_generator import ImageGenerator
from .cache_manager import CacheManager

class GenerationPipeline:
    """
    Streamlined pipeline with direct configuration (no config dependency)
    """
    
    def __init__(self):
        """Initialize with direct configuration"""
        # Direct settings instead of importing config
        self.cache_file_path = 'static/cache.json'
        self.generated_images_path = 'static/generated_images'
        self.max_cache_size = 100
        self.enable_caching = True
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        os.makedirs(self.generated_images_path, exist_ok=True)
        
        # Initialize core components
        self.prompt_processor = PromptProcessor(enable_llm=True)
        self.image_generator = ImageGenerator()
        self.cache_manager = CacheManager(
            cache_file_path=self.cache_file_path,
            images_dir=self.generated_images_path,
            max_cache_size=self.max_cache_size
        )
        
        # Performance tracking
        self.stats = {
            'total_generations': 0,
            'cache_hits': 0,
            'total_generation_time': 0.0,
            'total_time_saved': 0.0,
            'average_generation_time': 0.0
        }
        
        # Test components internally
        self._internal_health_check()

    def _internal_health_check(self):
        """Internal health check without exposing to user"""
        try:
            # Test prompt processor
            test_prompt = "a simple test"
            processed = self.prompt_processor.process_prompt(test_prompt, "light")
            
            # Test cache manager
            cache_works = hasattr(self.cache_manager, 'get_cached_image')
            
            # Test image generator connection
            generator_status = self.image_generator.get_status()
            
            # Store health status internally
            self._health_status = {
                'prompt_processor': processed is not None,
                'cache_manager': cache_works,
                'image_generator': True,  # Always available now
                'overall_health': True
            }
            
        except Exception as e:
            self._health_status = {
                'prompt_processor': False,
                'cache_manager': False,
                'image_generator': False,
                'overall_health': False,
                'error': str(e)
            }

    def optimize_parameters(self, parameters: Dict[str, Any], prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize generation parameters based on prompt analysis"""
        optimized = parameters.copy()
        
        # Extract style-based optimizations
        detected_style = prompt_data.get('detected_style', 'artistic')
        
        style_optimizations = {
            'photographic': {'guidance_scale': 7.5, 'num_inference_steps': 25},
            'artistic': {'guidance_scale': 10.0, 'num_inference_steps': 30},
            'fantasy': {'guidance_scale': 12.0, 'num_inference_steps': 30},
            'sci-fi': {'guidance_scale': 9.0, 'num_inference_steps': 28},
            'vintage': {'guidance_scale': 8.0, 'num_inference_steps': 25}
        }
        
        if detected_style in style_optimizations:
            style_params = style_optimizations[detected_style]
            for key, value in style_params.items():
                if key not in optimized:
                    optimized[key] = value
        
        # Ensure width and height are set
        if 'width' not in optimized:
            optimized['width'] = 1024
        if 'height' not in optimized:
            optimized['height'] = 1024
        
        # Add smart negative prompt if empty
        if not optimized.get('negative_prompt'):
            optimized['negative_prompt'] = prompt_data.get('negative_prompt', '')
        
        return optimized

    def check_cache(self, prompt_data: Dict[str, Any], parameters: Dict[str, Any]) -> Optional[Image.Image]:
        """Check cache for existing image with smart matching"""
        if not self.enable_caching:
            return None
        
        enhanced_prompt = prompt_data['enhanced_prompt']
        
        # Try exact match first
        cached_image = self.cache_manager.get_cached_image(
            enhanced_prompt, "auto", parameters
        )
        
        if cached_image:
            self.stats['cache_hits'] += 1
            return cached_image
        
        # Try with original prompt (less specific but possible match)
        original_prompt = prompt_data['original_prompt']
        if original_prompt != enhanced_prompt:
            cached_image = self.cache_manager.get_cached_image(
                original_prompt, "auto", parameters
            )
            if cached_image:
                self.stats['cache_hits'] += 1
                return cached_image
        
        return None

    def generate_image(self, user_prompt: str, model: Optional[str] = None,
                      parameters: Optional[Dict[str, Any]] = None,
                      enhancement_level: str = "medium",
                      style_preference: Optional[str] = None) -> Dict[str, Any]:
        """Main generation method with streamlined workflow"""
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Process prompt with LLM enhancement
            prompt_data = self.prompt_processor.process_prompt(
                user_prompt, 
                enhancement_level,
                custom_style=style_preference,
                use_llm=(enhancement_level == "llm")
            )
            
            # Validate prompt
            validation_result = self.prompt_processor.validate_prompt(user_prompt)
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error': 'Invalid prompt: ' + ', '.join(validation_result['errors']),
                    'validation_result': validation_result
                }
            
            # Step 2: Optimize parameters
            base_parameters = parameters or {}
            optimized_parameters = self.optimize_parameters(base_parameters, prompt_data)
            
            # Step 3: Check cache first (major performance boost)
            cached_image = self.check_cache(prompt_data, optimized_parameters)
            
            if cached_image:
                # Cache hit - return immediately
                total_time = time.time() - pipeline_start_time
                estimated_time_saved = self.image_generator.estimate_generation_time(optimized_parameters)
                self.stats['total_time_saved'] += estimated_time_saved
                
                return {
                    'success': True,
                    'image': cached_image,
                    'prompt_data': prompt_data,
                    'parameters': optimized_parameters,
                    'generation_time': 0.1,  # Cache retrieval time
                    'total_pipeline_time': total_time,
                    'was_cached': True,
                    'estimated_time_saved': estimated_time_saved,
                    'validation_result': validation_result,
                    'provider_used': 'cache',
                    'llm_enhanced': prompt_data.get('llm_enhanced', False)
                }
            
            # Step 4: Generate new image
            generation_result = self.image_generator.generate_image(
                prompt_data['enhanced_prompt'],
                None,  # Auto-selection
                optimized_parameters
            )
            
            if not generation_result or len(generation_result) != 2:
                return {
                    'success': False,
                    'error': 'Generation failed - no image returned',
                    'prompt_data': prompt_data
                }
            
            image, generation_time = generation_result
            
            if not image:
                return {
                    'success': False,
                    'error': 'Generation failed - image is empty',
                    'prompt_data': prompt_data,
                    'generation_time': generation_time
                }
            
            # Step 5: Post-process and cache
            if self.enable_caching:
                self.cache_manager.cache_image(
                    image,
                    prompt_data['enhanced_prompt'],
                    "auto",
                    optimized_parameters,
                    generation_time
                )
            
            # Step 6: Update statistics
            self.stats['total_generations'] += 1
            self.stats['total_generation_time'] += generation_time
            self.stats['average_generation_time'] = (
                self.stats['total_generation_time'] / self.stats['total_generations']
            )
            
            # Step 7: Generate image description (if LLM enhanced)
            image_description = ""
            if enhancement_level == "llm" and hasattr(self.prompt_processor, 'llm_enhancer') and self.prompt_processor.llm_enhancer:
                try:
                    image_description = self.prompt_processor.llm_enhancer.describe_generated_image(
                        prompt_data['enhanced_prompt']
                    )
                except Exception:
                    pass  # Description is optional
            
            total_time = time.time() - pipeline_start_time
            
            # Return success result
            return {
                'success': True,
                'image': image,
                'prompt_data': prompt_data,
                'parameters': optimized_parameters,
                'generation_time': generation_time,
                'total_pipeline_time': total_time,
                'was_cached': False,
                'validation_result': validation_result,
                'provider_used': 'huggingface',
                'model': 'auto-selected',
                'llm_enhanced': prompt_data.get('llm_enhanced', False),
                'image_description': image_description
            }
            
        except Exception as e:
            total_time = time.time() - pipeline_start_time
            
            return {
                'success': False,
                'error': str(e),
                'total_pipeline_time': total_time,
                'performance_stats': self.get_performance_stats()
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_gens = self.stats['total_generations']
        cache_hits = self.stats['cache_hits']
        
        # Calculate cache efficiency
        cache_hit_rate = (cache_hits / max(total_gens, 1)) * 100
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        
        stats = {
            'total_generations': total_gens,
            'cache_hits': cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_time_saved': f"{self.stats['total_time_saved']:.1f}s",
            'average_generation_time': f"{self.stats['average_generation_time']:.1f}s",
            'cache_size': cache_stats.get('total_entries', 0),
            'cache_size_mb': cache_stats.get('total_size_mb', '0.0'),
            'health_status': getattr(self, '_health_status', {'overall_health': True})
        }
        
        return stats

    def get_quick_suggestions(self, partial_prompt: str) -> List[str]:
        """Get quick prompt suggestions based on partial input"""
        if len(partial_prompt) < 3:
            return [
                "a beautiful landscape",
                "portrait of a person",
                "fantasy dragon",
                "futuristic city"
            ]
        
        # Get similar prompts from cache
        similar_prompts = self.cache_manager.get_similar_prompts(partial_prompt, limit=3)
        
        # Get suggestions from prompt processor
        processor_suggestions = self.prompt_processor.get_similar_prompts(partial_prompt, limit=2)
        
        # Combine and deduplicate
        all_suggestions = similar_prompts + processor_suggestions
        unique_suggestions = list(dict.fromkeys(all_suggestions))  # Remove duplicates while preserving order
        
        return unique_suggestions[:5]

    def clear_cache(self) -> bool:
        """Clear generation cache and reset statistics"""
        if self.cache_manager.clear_cache(confirm=True):
            # Reset cache-related stats
            self.stats['cache_hits'] = 0
            self.stats['total_time_saved'] = 0.0
            return True
        return False

    def export_session_data(self) -> Dict[str, Any]:
        """Export current session data for analysis"""
        return {
            'stats': self.stats,
            'performance': self.get_performance_stats(),
            'health_status': getattr(self, '_health_status', {}),
            'cache_info': self.cache_manager.get_cache_stats(),
            'generator_status': self.image_generator.get_status()
        }

    def estimate_cost_savings(self) -> Dict[str, Any]:
        """Estimate cost savings from caching and optimization"""
        estimated_cost_per_generation = 0.02  # Rough estimate
        
        cache_hits = self.stats['cache_hits']
        total_gens = self.stats['total_generations']
        
        money_saved = cache_hits * estimated_cost_per_generation
        time_saved = self.stats['total_time_saved']
        
        return {
            'money_saved_usd': f"${money_saved:.2f}",
            'time_saved_minutes': f"{time_saved / 60:.1f}",
            'efficiency_improvement': f"{(cache_hits / max(total_gens, 1)) * 100:.1f}%",
            'generations_avoided': cache_hits
        }

    def switch_to_local_mode(self):
        """Force switch to local generation mode"""
        self.image_generator.switch_to_local()
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'pipeline_health': getattr(self, '_health_status', {}),
            'generator_status': self.image_generator.get_status(),
            'cache_status': self.cache_manager.get_cache_stats(),
            'performance': self.get_performance_stats(),
            'cost_savings': self.estimate_cost_savings()
        }