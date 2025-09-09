import streamlit as st
import sys
import os
import io
import time
import json
from PIL import Image

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import freemium components
from src.sessions_manager import SessionManager
from src.freemium_ui_components import FreemiumUI
from src.freemium_image_generator import FreemiumImageGenerator

# Import your existing components
from src.prompt_processor import PromptProcessor
from src.cache_manager import CacheManager
from src.pipeline import GenerationPipeline
from src.utils import generate_filename, save_image

# Set page config
st.set_page_config(
    page_title="Generative LexiArt - AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.premium-badge {
    background: linear-gradient(45deg, #FFD700, #FFA500);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}

.free-badge {
    background: linear-gradient(45deg, #17a2b8, #007bff);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}

.visit-counter {
    background: rgba(255, 255, 255, 0.1);
    padding: 8px 15px;
    border-radius: 20px;
    color: white;
    font-size: 14px;
    margin-top: 10px;
}

.download-section {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
    border: 1px solid #e9ecef;
}

.generation-info {
    background: #e3f2fd;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #2196f3;
}

.upgrade-prompt {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin: 20px 0;
}

/* Ensure sidebar is always visible */
.stColumn > div {
    min-height: 100px;
}
</style>
""", unsafe_allow_html=True)

class VisitTracker:
    """Track total visits to the application"""
    
    def __init__(self):
        self.visits_file = "static/visits.json"
        self.ensure_directory()
        
    def ensure_directory(self):
        """Ensure the static directory exists"""
        os.makedirs("static", exist_ok=True)
    
    def get_visits(self) -> int:
        """Get current visit count"""
        try:
            if os.path.exists(self.visits_file):
                with open(self.visits_file, 'r') as f:
                    data = json.load(f)
                return data.get('total_visits', 0)
            return 0
        except:
            return 0
    
    def increment_visits(self) -> int:
        """Increment visit count and return new total"""
        current_visits = self.get_visits()
        new_visits = current_visits + 1
        
        try:
            data = {'total_visits': new_visits, 'last_updated': time.time()}
            with open(self.visits_file, 'w') as f:
                json.dump(data, f)
            return new_visits
        except:
            return current_visits

def create_device_download(image, device_type, prompt_data):
    """Create download button for specific device type"""
    
    # Define device-specific resolutions
    device_resolutions = {
        "phone": {"width": 1080, "height": 1920, "name": "Phone Wallpaper", "icon": "📱"},
        "desktop": {"width": 1920, "height": 1080, "name": "Desktop Wallpaper", "icon": "💻"}
    }
    
    if device_type not in device_resolutions:
        return
    
    target_res = device_resolutions[device_type]
    target_width = target_res["width"]
    target_height = target_res["height"]
    
    # Resize image for device
    resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Create download buffer
    img_buffer = io.BytesIO()
    resized_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    
    # Create filename
    timestamp = int(time.time())
    filename = f"lexiart_{device_type}_{target_width}x{target_height}_{timestamp}.png"
    
    # Create download button
    st.download_button(
        f"{target_res['icon']} Download for {target_res['name']} ({target_width}×{target_height})",
        img_buffer.getvalue(),
        filename,
        "image/png",
        use_container_width=True
    )

class AIImageStudio:
    """
    Main Generative LexiArt application with freemium integration
    FIXED: Always shows sidebar, even with upgrade prompts
    """
    
    def __init__(self):
        """Initialize the application with freemium support"""
        # Initialize visit tracking
        self.visit_tracker = VisitTracker()
        
        # Track visit if this is a new session
        if 'visit_tracked' not in st.session_state:
            self.total_visits = self.visit_tracker.increment_visits()
            st.session_state.visit_tracked = True
        else:
            self.total_visits = self.visit_tracker.get_visits()
        
        # Initialize freemium components
        self.session_manager = SessionManager()
        self.freemium_ui = FreemiumUI(self.session_manager)
        self.freemium_generator = FreemiumImageGenerator(self.session_manager)
        
        # Initialize your existing components
        self.prompt_processor = PromptProcessor(enable_llm=False)
        self.cache_manager = CacheManager()
        
    def run(self):
        """Main application runner - FIXED: Always shows sidebar"""
        self._show_header()
        
        # FIXED: Always show the main layout with sidebar
        main_col, sidebar_col = st.columns([2, 1])
        
        with main_col:
            self._show_main_content()
        
        with sidebar_col:
            self._show_sidebar()
    
    def _show_main_content(self):
        """Show main content area based on current state"""
        # Check if user wants to see premium setup (takes priority)
        if st.session_state.get('show_premium_setup', False):
            self._show_premium_setup_interface()
            return
        
        # Check if user needs to see upgrade prompt
        if self.freemium_ui.should_show_upgrade_prompt():
            self._show_upgrade_flow()
            return
        
        # Default: Show generation interface
        self._show_generation_interface()
    
    def _show_header(self):
        """Show application header with visit counter"""
        st.markdown(f"""
        <div class="main-header">
            <h1 style="margin: 0; color: white;">Generative LexiArt</h1>
            <p style="margin: 10px 0 0 0; color: white; font-size: 18px;">Transform your ideas into stunning visuals with AI</p>
            <div class="visit-counter">
                👥 Total Visits: {self.total_visits:,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_premium_setup_interface(self):
        """Show the premium setup interface"""
        self.freemium_ui.show_premium_setup_interface()
    
    def _show_upgrade_flow(self):
        """Show upgrade prompt in main content area"""
        # FIXED: Show upgrade prompt without hiding sidebar
        st.markdown("""
        <div class="upgrade-prompt">
            <h2 style="margin: 0 0 15px 0; color: white;">🎯 Free Trial Complete!</h2>
            <p style="margin: 0; font-size: 18px; color: white;">You've used all 5 free generations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🚀 Ready to unlock unlimited premium AI generation?**
        
        **Premium Features:**
        - ✨ **Unlimited** image generation
        - 🎨 **Professional quality** AI model
        - ⚡ **Faster** processing
        - 🎯 **Advanced** parameters
        - 🔥 **Higher resolution** support
        
        **How to upgrade:**
        1. 👉 **Look at the sidebar** (right side of screen)
        2. 🔍 Find the **"🌟 Premium Access"** section  
        3. 📝 Click **"🔑 Setup Premium Access"**
        4. 🆓 Get your **free** HuggingFace API key
        5. 🎉 Enjoy **unlimited** generations!
        """)
        
        # Add prominent button to draw attention to sidebar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("👉 Show Me How to Upgrade!", type="primary", use_container_width=True):
                st.session_state.show_premium_setup = True
                st.rerun()
        
        # Show some motivational content
        st.markdown("""
        ---
        **💡 Pro Tip:** Premium access is completely free! You just need a HuggingFace account 
        (also free) to get unlimited access to professional AI models.
        
        **🎨 What you can create with premium:**
        - Professional artwork and illustrations
        - High-quality wallpapers and backgrounds  
        - Creative designs for projects
        - Stunning digital art
        - And much more!
        """)
    
    def _show_generation_interface(self):
        """Show the main image generation interface"""
        # Show premium status if applicable
        self.freemium_ui.show_premium_status()
        
        st.subheader("✨ Create Your AI Masterpiece")
        
        # Prompt input
        prompt = st.text_area(
            "Describe your image:",
            height=120,
            placeholder="A magical forest with glowing trees, fantasy art style, detailed, high quality...",
            help="Be descriptive! Include style, mood, colors, and details for best results."
        )
        
        # Model and enhancement settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection (premium feature)
            available_models = self.freemium_generator.get_available_models()
            selected_model = self.freemium_ui.show_model_selector(available_models)
        
        with col2:
            # Enhancement level
            enhancement_level = st.selectbox(
                "Prompt Enhancement",
                ["light", "medium", "heavy"],
                index=1,
                help="How much to enhance your prompt automatically"
            )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options", expanded=False):
            st.markdown('<div class="generation-info">', unsafe_allow_html=True)
            st.info("📐 Generation Resolution: 1024×1024 (High Quality)")
            st.markdown('</div>', unsafe_allow_html=True)
            
            settings_col1, settings_col2 = st.columns(2)
            
            with settings_col1:
                max_steps = 50 if self.session_manager.is_premium_user() else 30
                steps = st.slider("Quality Steps", 10, max_steps, 20, help="Higher steps = better quality but slower")
                
            with settings_col2:
                guidance = st.slider("Creativity", 1.0, 20.0, 7.5, step=0.5, help="How closely to follow the prompt")
        
        # Generation button
        button_col, status_col = st.columns([3, 1])
        
        with button_col:
            can_generate = (prompt.strip() and 
                          (self.session_manager.can_generate_free() or self.session_manager.is_premium_user()))
            
            if not prompt.strip():
                button_text = "Enter a prompt first"
            elif not can_generate:
                button_text = "🔒 Check Sidebar to Upgrade"
            else:
                button_text = "🎨 Generate Image"
            
            generate_button = st.button(
                button_text,
                type="primary",
                disabled=not can_generate,
                use_container_width=True
            )
        
        with status_col:
            if self.session_manager.is_premium_user():
                st.markdown('<span class="premium-badge">⭐ Premium</span>', unsafe_allow_html=True)
            else:
                remaining = self.session_manager.get_remaining_free()
                st.markdown(f'<span class="free-badge">🎯 {remaining} left</span>', unsafe_allow_html=True)
        
        # Handle image generation
        if generate_button and prompt.strip():
            self._generate_image(prompt, selected_model, {
                'width': 1024,
                'height': 1024,
                'num_inference_steps': steps,
                'guidance_scale': guidance
            }, enhancement_level)
    
    def _generate_image(self, prompt, model, parameters, enhancement_level):
        """Handle the complete image generation process"""
        try:
            # Show generation info
            st.info("🎨 Generating 1024×1024 high-quality image...")
            
            # Step 1: Process prompt
            with st.spinner("🧠 Enhancing your prompt..."):
                processed_data = self.prompt_processor.process_prompt(prompt, enhancement_level)
            
            # Show enhanced prompt if different
            if processed_data['enhanced_prompt'] != prompt:
                with st.expander("✨ Enhanced Prompt", expanded=False):
                    st.write(f"**Original:** {prompt}")
                    st.write(f"**Enhanced:** {processed_data['enhanced_prompt']}")
            
            # Step 2: Check cache for existing image
            cached_image = self.cache_manager.get_cached_image(
                processed_data['enhanced_prompt'], model, parameters
            )
            
            if cached_image:
                st.success("⚡ Found in cache - instant result!")
                self._display_result(cached_image, processed_data, 0.1, True)
                return
            
            # Step 3: Generate new image using freemium generator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                progress_bar.progress(30)
                status_text.text("🎨 Creating your 1024×1024 masterpiece...")
                
                image, generation_time = self.freemium_generator.generate_image(
                    processed_data['enhanced_prompt'], model, parameters
                )
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if image:
                    # Step 4: Cache the result for future use
                    self.cache_manager.cache_image(
                        image, processed_data['enhanced_prompt'], model, parameters, generation_time
                    )
                    
                    # Step 5: Display the result
                    self._display_result(image, processed_data, generation_time, False)
                    
                    # Step 6: Handle post-generation messaging
                    self.freemium_ui.handle_post_generation_messaging(True)
                    
                else:
                    st.error("Generation failed. Please try again.")
                    self.freemium_ui.handle_post_generation_messaging(False)
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                raise e
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            self.freemium_ui.handle_post_generation_messaging(False)
    
    def _display_result(self, image, prompt_data, generation_time, was_cached):
        """Display the generated image with metadata and download options"""
        st.subheader("🎉 Your AI Creation")
        
        # Display the image
        st.image(image, use_column_width=True)
        
        # Show generation metadata
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Time:** {generation_time:.1f}s")
            st.write(f"**Cached:** {'Yes' if was_cached else 'No'}")
        with col2:
            mode = "Premium" if self.session_manager.is_premium_user() else "Free"
            st.write(f"**Mode:** {mode}")
            st.write(f"**Resolution:** {image.size[0]}×{image.size[1]}")
        
        # Download section with three options
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Download Options")
        
        # Original size download
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        filename = generate_filename(prompt_data['enhanced_prompt'])
        
        # Three download options in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                f"📄 Original ({image.size[0]}×{image.size[1]})",
                img_buffer.getvalue(),
                filename,
                "image/png",
                use_container_width=True
            )
        
        with col2:
            create_device_download(image, "phone", prompt_data)
        
        with col3:
            create_device_download(image, "desktop", prompt_data)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _show_sidebar(self):
        """FIXED: Always show sidebar with prominent upgrade option"""
        
        # ALWAYS show premium access first and prominently
        st.markdown("### 🌟 Upgrade to Premium")
        
        if not self.session_manager.is_premium_user():
            # Make the upgrade option very visible
            st.markdown("""
            <div style="
                background: linear-gradient(45deg, #28a745, #20c997);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <p style="color: white; margin: 0; font-weight: bold;">
                    🚀 Get Unlimited Access!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # BIG prominent button - FIXED: Unique key for sidebar
            if st.button("🔑 Setup Premium Access", type="primary", use_container_width=True, key="sidebar_main_premium_btn"):
                st.session_state.show_premium_setup = True
                st.rerun()
            
            st.markdown("""
            **✨ Premium includes:**
            - ♾️ Unlimited generations  
            - 🎨 Professional AI model
            - ⚡ Faster processing
            - 🆓 Completely FREE!
            """)
            
            # Show current usage status
            st.markdown("---")
            self.freemium_ui.show_free_trial_status()
        else:
            # Show premium status with sidebar context - FIXED: Added context
            self.freemium_ui.show_premium_status(context="sidebar")
        
        # Usage statistics
        st.markdown("---")
        self.freemium_ui.show_usage_stats()
        
        # Help and tips
        with st.expander("💡 Pro Tips", expanded=False):
            st.markdown("""
            **Writing Better Prompts:**
            - Be specific and descriptive
            - Include art style (oil painting, digital art)
            - Mention lighting and mood
            - Add quality keywords
            
            **Examples:**
            - "Sunset over mountains, oil painting style"
            - "Cyberpunk city, neon lights, detailed"
            - "Cute dragon, cartoon style, colorful"
            
            **Resolution Info:**
            - All images generate at 1024×1024 for best quality
            - Download optimized versions for phone/desktop
            - Premium users get additional AI models
            """)
        
        # Performance statistics
        with st.expander("📊 Performance Stats", expanded=False):
            cache_stats = self.cache_manager.get_cache_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cached Images", cache_stats.get('total_entries', 0))
                st.metric("Cache Hits", cache_stats.get('total_cache_hits', 0))
            
            with col2:
                st.metric("Time Saved", f"{cache_stats.get('total_generation_time_saved', 0)}s")
                st.metric("Cache Size", f"{cache_stats.get('total_size_mb', 0)} MB")
        
        # Debug info (optional - can be removed) - FIXED: Unique key
        if st.button("🔧 Debug Session", help="Show session debug info", key="debug_session_sidebar"):
            debug_info = self.session_manager.debug_session_info()
            st.json(debug_info)


def main():
    """Main application entry point"""
    try:
        app = AIImageStudio()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details (for debugging)"):
            st.exception(e)


if __name__ == "__main__":
    main()
