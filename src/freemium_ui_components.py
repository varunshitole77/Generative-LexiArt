import streamlit as st
from typing import Dict, Any

class FreemiumUI:
    """Updated UI components highlighting Stability AI premium access"""
    
    def __init__(self, session_manager):
        """Initialize with session manager"""
        self.session_manager = session_manager
    
    def show_welcome_banner(self):
        """Show welcome banner for new users"""
        if self.session_manager.get_remaining_free() == 5:
            st.markdown("""
            <div style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 20px;
            ">
                <h2 style="margin: 0; color: white;">Welcome to AI Image Studio!</h2>
                <p style="margin: 10px 0 0 0; color: white;">Generate 5 stunning AI images completely FREE - no signup required!</p>
                <p style="margin: 5px 0 0 0; color: white; font-size: 14px;">Premium: Unlock professional AI model</p>
            </div>
            """, unsafe_allow_html=True)
    
    def show_free_trial_status(self):
        """Show current free trial status with premium upgrade hint"""
        remaining = self.session_manager.get_remaining_free()
        used = st.session_state.free_images_used
        
        if remaining > 0:
            progress = used / 5
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(progress)
            with col2:
                st.write(f"**{remaining} left**")
            
            if remaining <= 2:
                st.warning(f"{remaining} free generations remaining!")
                st.info("Upgrade to unlock premium AI model with unlimited generations")
            else:
                st.success(f"{remaining} free AI generations available")
        else:
            st.error("Free trial completed!")
            st.info("Use Premium Setup in sidebar to access premium AI model")
    
    def show_upgrade_prompt(self):
        """Show premium upgrade prompt"""
        if not self.session_manager.can_generate_free() and not self.session_manager.is_premium_user():
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 20px 0;
            ">
                <h2 style="margin: 0 0 15px 0; color: white;">Unlock Premium AI Generation</h2>
                <p style="margin: 0; font-size: 18px; color: white;">Access professional AI model with premium!</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🚀 Premium Features:**
                - **Professional quality** generation
                - **Unlimited** generations
                - **Advanced AI** model
                - **Priority** processing
                - **Higher resolution** support
                - **Custom parameters**
                """)
            
            with col2:
                st.markdown("""
                **🎯 Why Go Premium?**
                - Industry-leading quality
                - Proven reliability
                - Optimized performance
                - Professional results
                - Latest AI technology
                - No generation limits
                """)
            
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
                border-left: 5px solid #28a745;
            ">
                <h3 style="color: #28a745; margin: 0 0 10px 0;">Setup Premium Access</h3>
                <p style="margin: 0; color: #6c757d;">Use Premium Setup in sidebar for exclusive AI model!</p>
            </div>
            """, unsafe_allow_html=True)
            
            return True
        return False
    
    def show_premium_setup_interface(self):
        """Show the premium setup interface"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin: 20px 0;
        ">
            <h2 style="margin: 0 0 15px 0; color: white;">Premium AI Setup</h2>
            <p style="margin: 0; font-size: 18px; color: white;">Unlock professional AI model with a free API key!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show premium benefits
        with st.expander("🎨 Premium Benefits", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🆓 Free to Start:**
                - No credit card required
                - Free HuggingFace account
                - Instant activation
                
                **⚡ Performance:**
                - Fast generation
                - Reliable service
                - Professional quality
                """)
            
            with col2:
                st.markdown("""
                **🎯 Features:**
                - Unlimited generations
                - Advanced parameters
                - Priority processing
                
                **🤖 AI Technology:**
                - Latest AI model
                - Optimized quality
                - Smart processing
                """)
        
        # Instructions
        with st.expander("📋 Setup Instructions", expanded=True):
            st.markdown("""
            **How to unlock premium AI model:**
            
            1. **Visit HuggingFace**: Click the button below
            2. **Create Account**: Free signup (no credit card needed)
            3. **Go to Settings**: Profile → Settings → Access Tokens
            4. **Create Token**: Click "New token" with "Read" permission
            5. **Copy Token**: Copy the token starting with "hf_"
            6. **Paste Below**: Enter token to unlock premium features
            7. **Generate**: Create unlimited images with premium model!
            
            **Why use premium?**
            - Professional quality results
            - Unlimited image generation
            - Advanced AI technology
            - Reliable and fast processing
            """)
        
        # Direct link to HuggingFace
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <a href="https://huggingface.co/settings/tokens" target="_blank" style="
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                color: white;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                font-size: 18px;
                display: inline-block;
            ">
                🔗 Get Free API Key
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input
        self.show_api_key_input()
        
        # Close button
        if st.button("❌ Close Premium Setup", use_container_width=True):
            st.session_state.show_premium_setup = False
            st.rerun()
    
    def show_api_key_input(self):
        """Show API key input interface"""
        st.subheader("🔐 Enter Your HuggingFace API Key")
        
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            help="Paste your HuggingFace API key to unlock premium AI model"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("✅ Unlock Premium Access", type="primary", disabled=not api_key_input, use_container_width=True):
                if api_key_input:
                    if self.session_manager.set_premium_mode(api_key_input):
                        st.success("🎉 Premium AI model activated! Unlimited generations unlocked!")
                        st.balloons()
                        st.session_state.show_premium_setup = False
                        st.rerun()
                    else:
                        st.error("❌ Invalid API key. Please check and try again.")
                        st.info("Make sure your key starts with 'hf_' and is from HuggingFace")
                else:
                    st.warning("Please enter your API key")
        
        with col2:
            if st.button("❓ Need Help?", use_container_width=True):
                st.info("Visit: https://huggingface.co/settings/tokens")
    
    def show_premium_status(self):
        """Show premium user status"""
        if self.session_manager.is_premium_user():
            st.markdown("""
            <div style="
                background: linear-gradient(45deg, #28a745, #20c997);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 20px;
            ">
                <h3 style="margin: 0; color: white;">🌟 Premium Active - Unlimited Generations!</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Premium controls
            with st.expander("⚙️ Premium Settings", expanded=False):
                st.markdown("""
                **🎨 Your Premium Features:**
                - Professional AI model
                - Unlimited generations
                - Advanced parameters
                - Priority processing
                - Higher quality output
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Change API Key", use_container_width=True):
                        st.session_state.show_premium_setup = True
                        st.rerun()
                
                with col2:
                    # Option to switch back to free mode
                    if st.button("⬇️ Switch to Free Mode", use_container_width=True):
                        if st.session_state.get('confirm_switch_to_free'):
                            # Actually switch
                            st.session_state.api_mode = 'free'
                            st.session_state.huggingface_api_key = ''
                            st.session_state.confirm_switch_to_free = False
                            self.session_manager.save_usage_count()
                            st.success("Switched back to free mode!")
                            st.rerun()
                        else:
                            # Ask for confirmation
                            st.session_state.confirm_switch_to_free = True
                            st.rerun()
                
                # Show confirmation if needed
                if st.session_state.get('confirm_switch_to_free'):
                    st.warning("⚠️ Are you sure? You'll lose access to premium features.")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Yes, Switch to Free", type="primary"):
                            st.session_state.api_mode = 'free'
                            st.session_state.huggingface_api_key = ''
                            st.session_state.confirm_switch_to_free = False
                            self.session_manager.save_usage_count()
                            st.success("Switched back to free mode!")
                            st.rerun()
                    with confirm_col2:
                        if st.button("❌ Keep Premium"):
                            st.session_state.confirm_switch_to_free = False
                            st.rerun()
                
                # Show current key info
                if st.session_state.huggingface_api_key:
                    masked_key = st.session_state.huggingface_api_key[:8] + "..." + st.session_state.huggingface_api_key[-4:]
                    st.info(f"🔐 Current API Key: {masked_key}")
    
    def show_model_selector(self, available_models):
        """Hidden model selector - always returns auto"""
        # Model selection is now hidden from users
        return "auto"
    
    def should_show_upgrade_prompt(self) -> bool:
        """Check if upgrade prompt should be shown"""
        return ((not self.session_manager.can_generate_free() and 
                not self.session_manager.is_premium_user()) or
                st.session_state.get('show_premium_setup', False))
    
    def handle_post_generation_messaging(self, was_successful: bool):
        """Handle messaging after image generation"""
        if was_successful:
            remaining = self.session_manager.get_remaining_free()
            
            if self.session_manager.is_premium_user():
                st.success("🌟 Premium generation complete! Create unlimited more.")
            elif remaining == 0:
                st.warning("🎯 That was your last free generation! Unlock premium in sidebar.")
            elif remaining <= 2:
                st.info(f"⚠️ {remaining} free generations left. Consider premium for unlimited access.")
    
    def show_usage_stats(self):
        """Show usage statistics"""
        stats = self.session_manager.get_user_stats()
        
        with st.expander("📊 Usage Statistics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if stats['is_premium']:
                    st.metric("Mode", "🌟 Premium")
                    st.metric("Status", "Unlimited")
                else:
                    st.metric("Mode", "🆓 Free Trial")
                    st.metric("Free Images Used", stats['free_images_used'])
            
            with col2:
                if stats['is_premium']:
                    st.metric("Generations", "♾️ Unlimited")
                    st.metric("Quality", "🏆 Professional")
                else:
                    st.metric("Remaining", stats['free_remaining'])
                    st.metric("Unlock", "Premium Access")
    
    def show_sidebar_premium_access(self):
        """Show premium access in sidebar"""
        if not self.session_manager.is_premium_user():
            with st.expander("🌟 Premium Access", expanded=True):
                st.markdown("""
                **🚀 Unlock Premium:**
                - 🏆 **Professional** quality
                - ♾️ **Unlimited** generations
                - ⚡ **Fast** processing
                - 🎯 **Advanced** features
                - 🤖 **Latest AI** technology
                """)
                
                # THE ONLY premium setup button
                if st.button("🔐 Setup Premium Access", type="primary", use_container_width=True):
                    st.session_state.show_premium_setup = True
                    st.rerun()
                
                st.markdown("""
                <div style="text-align: center; margin: 10px 0;">
                    <small style="color: #666;">
                        Free HuggingFace account • Professional AI model
                    </small>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show current premium status
            with st.expander("🌟 Premium Active", expanded=False):
                st.markdown("""
                **Your Premium Features:**
                - 🏆 Professional quality
                - ♾️ Unlimited generations
                - ⚡ Priority processing
                - 🎯 Advanced parameters
                - 🤖 Latest AI model
                
                **Status:** ♾️ Unlimited generations
                """)
    
    def show_stability_ai_info_panel(self):
        """Hidden - no longer showing model info"""
        pass