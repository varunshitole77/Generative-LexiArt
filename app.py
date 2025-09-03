"""
Generative LexiArt - Main Streamlit Application
Advanced text-to-image generation system with optimized processing pipeline
"""

import streamlit as st
import os
from datetime import datetime
from PIL import Image
import io

# Import our custom modules
from src.pipeline import GenerationPipeline
from src.prompt_processor import PromptProcessor
from src.utils import format_generation_time, get_file_size
from config.settings import get_settings, validate_api_key

# Page configuration
st.set_page_config(
    page_title="Generative LexiArt",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .generation-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

def setup_pipeline():
    """Setup the generation pipeline"""
    if st.session_state.pipeline is None:
        with st.spinner("🔧 Initializing generation pipeline..."):
            try:
                # Try different generation methods in order of preference
                generation_method = None
                
                # Check if local generation is available
                try:
                    import diffusers
                    import torch
                    generation_method = 'local'
                    st.info("🏠 Using local generation (unlimited & free!)")
                except ImportError:
                    st.info("💡 For unlimited free generation, install: pip install diffusers torch")
                
                # Fallback to API if local not available
                if generation_method is None:
                    if validate_api_key('huggingface'):
                        generation_method = 'huggingface'
                        st.info("🌐 Using Hugging Face API")
                    else:
                        st.error("❌ No generation method available!")
                        st.info("**Options:**")
                        st.info("1. Install local generation: `pip install diffusers torch`")
                        st.info("2. Fix your Hugging Face API token")
                        st.stop()
                
                st.session_state.pipeline = GenerationPipeline(generation_method)
                st.success("✅ Pipeline initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize pipeline: {str(e)}")
                st.info("**Troubleshooting:**")
                st.info("1. Install dependencies: `pip install diffusers torch`")
                st.info("2. Check your internet connection")
                st.info("3. Try restarting the app")
                st.stop()

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>Generative LexiArt</h1>
        <p>Text-to-Image Generation using LLMs</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with controls and information"""
    st.sidebar.title("⚙️ Generation Settings")
    
    # Show current generation method
    if st.session_state.pipeline:
        method = st.session_state.pipeline.api_type
        if method == 'local':
            st.sidebar.success("🏠 Local Generation (Unlimited & Free!)")
        elif method == 'huggingface':
            st.sidebar.info("🌐 Hugging Face API")
        else:
            st.sidebar.info(f"🔧 Using: {method}")
    
    # Model selection
    try:
        available_models = st.session_state.pipeline.image_generator.get_available_models()
        selected_model = st.sidebar.selectbox(
            "🤖 Select Model",
            available_models,
            index=0,
            help="Choose the AI model for image generation"
        )
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        selected_model = "runwayml/stable-diffusion-v1-5"  # Fallback
    
    # Image parameters
    st.sidebar.subheader("🖼️ Image Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
    with col2:
        height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)
    
    steps = st.sidebar.slider(
        "Inference Steps", 
        min_value=10, max_value=100, value=20, step=5,
        help="More steps = better quality but slower generation"
    )
    
    guidance_scale = st.sidebar.slider(
        "Guidance Scale", 
        min_value=1.0, max_value=20.0, value=7.5, step=0.5,
        help="How closely to follow the prompt (higher = more strict)"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        enhancement_level = st.selectbox(
            "Prompt Enhancement",
            ["light", "medium", "heavy"],
            index=1,
            help="Level of automatic prompt improvement"
        )
        
        use_negative_prompt = st.checkbox("Use Negative Prompt", value=True)
        
        if use_negative_prompt:
            custom_negative = st.text_area(
                "Custom Negative Prompt",
                placeholder="Enter things to avoid in the image...",
                height=100
            )
        else:
            custom_negative = ""
        
        seed = st.number_input(
            "Seed (-1 for random)", 
            min_value=-1, max_value=999999999, value=-1,
            help="Use specific seed for reproducible results"
        )
    
    # Performance stats
    st.sidebar.subheader("📊 Performance Stats")
    if st.session_state.pipeline:
        stats = st.session_state.pipeline.get_performance_stats()
        
        st.sidebar.metric("Total Generations", stats['total_generations'])
        st.sidebar.metric("Cache Hit Rate", stats['cache_hit_rate'])
        st.sidebar.metric("Time Saved", stats['total_time_saved'])
    
    # Cache management
    st.sidebar.subheader("💾 Cache Management")
    if st.sidebar.button("📊 View Cache Stats"):
        st.session_state.show_cache_stats = True
    
    if st.sidebar.button("🗑️ Clear Cache"):
        if st.session_state.pipeline.clear_cache():
            st.sidebar.success("Cache cleared!")
        else:
            st.sidebar.error("Failed to clear cache")
    
    return {
        'model': selected_model,
        'width': width,
        'height': height,
        'num_inference_steps': steps,
        'guidance_scale': guidance_scale,
        'enhancement_level': enhancement_level,
        'negative_prompt': custom_negative,
        'seed': seed if seed != -1 else None
    }

def display_prompt_input():
    """Display prompt input section"""
    st.subheader("✍️ Create Your Image")
    
    # Prompt templates
    with st.expander("📋 Quick Start Templates"):
        templates = st.session_state.pipeline.prompt_processor.get_prompt_templates()
        
        col1, col2 = st.columns(2)
        template_cols = [col1, col2]
        
        for i, (name, template) in enumerate(templates.items()):
            with template_cols[i % 2]:
                if st.button(f"📝 {name}", key=f"template_{name}"):
                    subject = st.text_input(f"Subject for {name}", key=f"subject_{name}")
                    if subject:
                        applied_template = st.session_state.pipeline.prompt_processor.apply_template(template, subject)
                        st.session_state.template_prompt = applied_template
    
    # Main prompt input
    default_prompt = getattr(st.session_state, 'template_prompt', '')
    user_prompt = st.text_area(
        "🎨 Describe your image:",
        value=default_prompt,
        height=100,
        placeholder="Enter your creative prompt here... e.g., 'a magical forest with glowing mushrooms, fantasy art style, detailed, high quality'"
    )
    
    # Prompt enhancement preview
    if user_prompt.strip():
        with st.expander("🔍 Prompt Enhancement Preview"):
            try:
                preview_data = st.session_state.pipeline.prompt_processor.process_prompt(
                    user_prompt, "medium"
                )
                
                st.write("**Original Prompt:**")
                st.text(preview_data['original_prompt'])
                
                st.write("**Enhanced Prompt:**")
                st.text(preview_data['enhanced_prompt'])
                
                st.write("**Detected Style:**", preview_data['detected_style'])
                
                if preview_data['suggestions']:
                    st.write("**Suggestions for Improvement:**")
                    for suggestion in preview_data['suggestions']:
                        st.write(f"• {suggestion}")
                        
            except Exception as e:
                st.warning(f"Preview error: {str(e)}")
    
    return user_prompt

def display_generation_results(result):
    """Display generation results"""
    if result['success']:
        image = result['image']
        
        # Display image
        st.image(image, caption="Generated Image", use_column_width=True)
        
        # Generation info
        st.markdown('<div class="generation-info">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Generation Time", f"{result['generation_time']:.2f}s")
        
        with col2:
            st.metric("Pipeline Time", f"{result['total_pipeline_time']:.2f}s")
        
        with col3:
            cache_status = "✅ Cached" if result['was_cached'] else "🆕 Generated"
            st.metric("Source", cache_status)
        
        with col4:
            if result.get('saved_filepath'):
                file_size = get_file_size(result['saved_filepath'])
                st.metric("File Size", file_size)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        st.download_button(
            label="💾 Download Image",
            data=img_bytes,
            file_name=f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
        
        # Detailed information
        with st.expander("ℹ️ Generation Details"):
            prompt_data = result['prompt_data']
            
            st.write("**Model:**", result['model'])
            st.write("**Enhancement Level:**", prompt_data.get('enhancement_level', 'N/A'))
            st.write("**Detected Style:**", prompt_data.get('detected_style', 'N/A'))
            
            st.write("**Parameters Used:**")
            params = result['parameters']
            for key, value in params.items():
                st.write(f"• {key}: {value}")
            
            if prompt_data.get('negative_prompt'):
                st.write("**Negative Prompt:**")
                st.text(prompt_data['negative_prompt'])
        
        # Add to session history
        st.session_state.generation_history.append({
            'timestamp': datetime.now(),
            'prompt': result['prompt_data']['original_prompt'],
            'enhanced_prompt': result['prompt_data']['enhanced_prompt'],
            'image': image,
            'generation_time': result['generation_time'],
            'was_cached': result['was_cached'],
            'model': result['model']
        })
        
        st.session_state.current_image = image
        
    else:
        st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        
        if 'prompt_data' in result:
            st.write("**Processed prompt data was:**")
            st.json(result['prompt_data'])

def display_history():
    """Display generation history"""
    if st.session_state.generation_history:
        st.subheader("📚 Generation History")
        
        for i, item in enumerate(reversed(st.session_state.generation_history[-10:])):  # Show last 10
            with st.expander(f"🎨 {item['timestamp'].strftime('%H:%M:%S')} - {item['prompt'][:50]}..."):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(item['image'], width=200)
                
                with col2:
                    st.write(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Model:** {item['model']}")
                    st.write(f"**Generation Time:** {item['generation_time']:.2f}s")
                    st.write(f"**Source:** {'Cache' if item['was_cached'] else 'Generated'}")
                    st.write("**Original Prompt:**")
                    st.text(item['prompt'])

def display_cache_stats():
    """Display detailed cache statistics"""
    if hasattr(st.session_state, 'show_cache_stats') and st.session_state.show_cache_stats:
        st.subheader("💾 Cache Statistics")
        
        stats = st.session_state.pipeline.cache_manager.get_cache_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", stats['total_entries'])
        
        with col2:
            st.metric("Cache Size", f"{stats['total_size_mb']} MB")
        
        with col3:
            st.metric("Total Cache Hits", stats['total_cache_hits'])
        
        with col4:
            st.metric("Cache Efficiency", stats['cache_efficiency'])
        
        st.write(f"**Time Saved:** {stats['total_generation_time_saved']:.1f} seconds")
        
        if stats['oldest_entry']:
            st.write(f"**Oldest Entry:** {stats['oldest_entry']}")
        if stats['newest_entry']:
            st.write(f"**Newest Entry:** {stats['newest_entry']}")
        
        # Reset the flag
        st.session_state.show_cache_stats = False

def display_tips():
    """Display usage tips and tricks"""
    with st.sidebar.expander("💡 Tips & Tricks"):
        st.markdown("""
        **🎨 For Better Results:**
        - Be specific with details
        - Include art style keywords
        - Mention lighting and colors
        
        **⚡ Performance Tips:**
        - Similar prompts use cache automatically
        - Lower steps = faster generation
        - Higher guidance = more prompt adherence
        
        **🔧 Advanced Usage:**
        - Use negative prompts to avoid unwanted elements
        - Set specific seeds for reproducible results
        - Try different enhancement levels
        """)

def test_pipeline():
    """Test pipeline functionality"""
    st.subheader("🧪 Pipeline Test")
    
    if st.button("🚀 Run Pipeline Test"):
        with st.spinner("Testing pipeline..."):
            test_result = st.session_state.pipeline.test_pipeline()
            
            if test_result['success']:
                st.success("✅ Pipeline test successful!")
                st.image(test_result['image'], caption="Test Generation", width=300)
            else:
                st.error(f"❌ Test failed: {test_result.get('error')}")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Setup pipeline
    setup_pipeline()
    
    # Display sidebar controls
    generation_params = display_sidebar()
    
    # Display tips
    display_tips()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prompt input section
        user_prompt = display_prompt_input()
        
        # Generate button
        if st.button("🎨 Generate Image", type="primary", disabled=not user_prompt.strip()):
            if not user_prompt.strip():
                st.warning("⚠️ Please enter a prompt to generate an image.")
            else:
                with st.spinner("🎨 Creating your masterpiece..."):
                    # Prepare parameters
                    params = {
                        'width': generation_params['width'],
                        'height': generation_params['height'],
                        'num_inference_steps': generation_params['num_inference_steps'],
                        'guidance_scale': generation_params['guidance_scale']
                    }
                    
                    # Add negative prompt if specified
                    if generation_params['negative_prompt']:
                        params['negative_prompt'] = generation_params['negative_prompt']
                    
                    # Add seed if specified
                    if generation_params['seed'] is not None:
                        params['seed'] = generation_params['seed']
                    
                    # Generate image
                    result = st.session_state.pipeline.generate_image(
                        user_prompt,
                        model=generation_params['model'],
                        parameters=params,
                        enhancement_level=generation_params['enhancement_level']
                    )
                    
                    # Display results
                    display_generation_results(result)
        
        # Display current image if available
        if st.session_state.current_image:
            st.subheader("🖼️ Current Generation")
            st.image(st.session_state.current_image, use_column_width=True)
    
    with col2:
        # Performance dashboard
        st.subheader("⚡ Performance Dashboard")
        
        if st.session_state.pipeline:
            perf_stats = st.session_state.pipeline.get_performance_stats()
            
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            
            st.metric("Cache Hit Rate", perf_stats['cache_hit_rate'])
            st.metric("Performance Gain", perf_stats['performance_improvement'])
            st.metric("Average Gen Time", perf_stats['average_generation_time'])
            
            # Progress bar for cache hit rate
            cache_hit_rate = perf_stats['cache_hit_rate']
            if isinstance(cache_hit_rate, str):
                cache_rate = float(cache_hit_rate.rstrip('%'))
            else:
                cache_rate = float(cache_hit_rate)
            st.progress(min(cache_rate / 100, 1.0))
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        if st.button("🎲 Random Art Style"):
            styles = ["digital art", "oil painting", "watercolor", "photography", "concept art", "fantasy art"]
            import random
            random_style = random.choice(styles)
            st.info(f"Try adding: '{random_style}' to your prompt!")
        
        if st.button("✨ Enhance Last Prompt"):
            if st.session_state.generation_history:
                last_prompt = st.session_state.generation_history[-1]['prompt']
                enhanced = st.session_state.pipeline.prompt_processor.process_prompt(last_prompt, "heavy")
                st.text_area("Enhanced Version:", enhanced['enhanced_prompt'], height=100)
    
    # Cache statistics (if requested)
    display_cache_stats()
    
    # Generation history
    display_history()
    
    # Development/Testing section (can be hidden in production)
    with st.expander("🔧 Development Tools"):
        test_pipeline()
        
        if st.button("📊 Show Detailed Performance Stats"):
            if st.session_state.pipeline:
                detailed_stats = st.session_state.pipeline.get_performance_stats()
                st.json(detailed_stats)
        
        if st.button("🔍 Show Pipeline State"):
            st.write("**Session State Keys:**", list(st.session_state.keys()))
            st.write("**Generation History Count:**", len(st.session_state.generation_history))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p> Generative LexiArt | Powered by Advanced AI & Optimized Processing Pipeline</p>
        <p>Built with Streamlit • LangChain • Hugging Face • Diffusion Models</p>
    </div>
    """, unsafe_allow_html=True)

# Error handling for the entire app
def run_app():
    """Run the app with error handling"""
    try:
        main()
    except Exception as e:
        st.error("🚨 Application Error")
        st.exception(e)
        
        st.info("**Troubleshooting Steps:**")
        st.write("1. Check your .env file has the correct API key")
        st.write("2. Ensure all dependencies are installed")
        st.write("3. Try refreshing the page")
        st.write("4. Check your internet connection")

if __name__ == "__main__":
    run_app()