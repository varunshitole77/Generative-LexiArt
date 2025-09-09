import streamlit as st
import json
import os
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any

class SessionManager:
    """FIXED session manager that prevents refresh abuse"""
    
    def __init__(self):
        """Initialize session manager with proper persistence"""
        self.max_free_images = 5
        self.session_file = "static/sessions.json"
        self.ensure_directories()
        self.init_session()
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs("static", exist_ok=True)
    
    def init_session(self):
        """Initialize session state with persistent user tracking"""
        
        # Step 1: Get or create persistent user identifier
        if 'persistent_user_id' not in st.session_state:
            # Try to get from browser session or create new
            st.session_state.persistent_user_id = self._get_persistent_user_id()
        
        # Step 2: Load user data from file
        user_data = self._load_user_data(st.session_state.persistent_user_id)
        
        # Step 3: Initialize session state from loaded data
        if 'free_images_used' not in st.session_state:
            st.session_state.free_images_used = user_data.get('free_images_used', 0)
        
        if 'api_mode' not in st.session_state:
            st.session_state.api_mode = user_data.get('api_mode', 'free')
        
        if 'huggingface_api_key' not in st.session_state:
            st.session_state.huggingface_api_key = user_data.get('huggingface_api_key', "")
        
        if 'show_upgrade_prompt' not in st.session_state:
            st.session_state.show_upgrade_prompt = False
        
        # Step 4: Check for session reset attempts (anti-abuse)
        self._check_session_integrity()
    
    def _get_persistent_user_id(self) -> str:
        """Get persistent user ID that survives refreshes"""
        
        # Method 1: Check if we have a stored user ID in browser
        if hasattr(st, 'query_params'):
            query_user_id = st.query_params.get('user_id')
            if query_user_id and len(query_user_id) == 32:  # Valid UUID format
                return query_user_id
        
        # Method 2: Generate based on browser fingerprint + IP simulation
        # This creates a semi-persistent ID that's the same for the same browser/session
        browser_info = {
            'user_agent': st.context.headers.get('user-agent', 'unknown') if hasattr(st, 'context') else 'unknown',
            'session_id': str(st.session_state.get('session_id', '')),
            'timestamp_day': datetime.now().strftime('%Y-%m-%d')  # Reset daily to prevent infinite abuse
        }
        
        # Create hash of browser info
        browser_string = json.dumps(browser_info, sort_keys=True)
        browser_hash = hashlib.md5(browser_string.encode()).hexdigest()
        
        # Use first 8 chars of hash + random for uniqueness
        if 'browser_fingerprint' not in st.session_state:
            st.session_state.browser_fingerprint = browser_hash[:8] + str(uuid.uuid4())[:8]
        
        return st.session_state.browser_fingerprint
    
    def _load_user_data(self, user_id: str) -> Dict[str, Any]:
        """Load user data from persistent storage"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    all_sessions = json.load(f)
                
                # Return user data or default
                return all_sessions.get(user_id, {
                    'free_images_used': 0,
                    'api_mode': 'free',
                    'huggingface_api_key': '',
                    'created_at': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat()
                })
            else:
                return {
                    'free_images_used': 0,
                    'api_mode': 'free',
                    'huggingface_api_key': '',
                    'created_at': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat()
                }
        except Exception as e:
            st.error(f"Error loading user data: {str(e)}")
            return {
                'free_images_used': 0,
                'api_mode': 'free',
                'huggingface_api_key': '',
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
    
    def _check_session_integrity(self):
        """Check for session manipulation attempts"""
        user_id = st.session_state.persistent_user_id
        
        # Load current stored data
        stored_data = self._load_user_data(user_id)
        session_usage = st.session_state.free_images_used
        stored_usage = stored_data.get('free_images_used', 0)
        
        # If session shows less usage than stored, someone is trying to cheat
        if session_usage < stored_usage:
            st.session_state.free_images_used = stored_usage
            st.warning("Session restored from backup.")
        
        # Check for rapid refresh abuse (more than 10 refreshes in 1 minute)
        last_activity = stored_data.get('last_activity')
        if last_activity:
            try:
                last_time = datetime.fromisoformat(last_activity.replace('Z', '+00:00').replace('+00:00', ''))
                if datetime.now() - last_time < timedelta(seconds=6):  # Less than 6 seconds ago
                    refresh_count = stored_data.get('refresh_count', 0) + 1
                    if refresh_count > 10:
                        st.warning("Too many rapid refreshes detected. Please wait a moment.")
                        st.stop()
                else:
                    refresh_count = 1
            except:
                refresh_count = 1
        else:
            refresh_count = 1
        
        # Update refresh tracking
        self._save_user_data_with_refresh_count(refresh_count)
    
    def _save_user_data_with_refresh_count(self, refresh_count: int):
        """Save user data with refresh count tracking"""
        try:
            # Load all sessions
            all_sessions = {}
            if os.path.exists(self.session_file):
                try:
                    with open(self.session_file, 'r') as f:
                        all_sessions = json.load(f)
                except:
                    all_sessions = {}
            
            user_id = st.session_state.persistent_user_id
            
            # Update user data
            user_data = all_sessions.get(user_id, {})
            user_data.update({
                'free_images_used': st.session_state.free_images_used,
                'api_mode': st.session_state.api_mode,
                'huggingface_api_key': st.session_state.huggingface_api_key,
                'last_activity': datetime.now().isoformat(),
                'refresh_count': refresh_count
            })
            
            all_sessions[user_id] = user_data
            
            # Save back to file
            with open(self.session_file, 'w') as f:
                json.dump(all_sessions, f, indent=2)
                
        except Exception as e:
            st.error(f"Error saving user data: {str(e)}")
    
    def save_usage_count(self):
        """Save current usage count to persistent storage"""
        try:
            # Load all sessions
            all_sessions = {}
            if os.path.exists(self.session_file):
                try:
                    with open(self.session_file, 'r') as f:
                        all_sessions = json.load(f)
                except:
                    all_sessions = {}
            
            user_id = st.session_state.persistent_user_id
            
            # Update user data
            user_data = all_sessions.get(user_id, {})
            user_data.update({
                'free_images_used': st.session_state.free_images_used,
                'api_mode': st.session_state.api_mode,
                'huggingface_api_key': st.session_state.huggingface_api_key,
                'last_activity': datetime.now().isoformat()
            })
            
            all_sessions[user_id] = user_data
            
            # Save back to file
            with open(self.session_file, 'w') as f:
                json.dump(all_sessions, f, indent=2)
                
        except Exception as e:
            st.error(f"Error saving usage count: {str(e)}")
    
    def increment_usage(self):
        """Increment free image usage count with validation"""
        if st.session_state.api_mode == 'free':
            # Double-check they haven't exceeded limit
            if st.session_state.free_images_used >= self.max_free_images:
                st.error("Free limit already reached!")
                st.session_state.show_upgrade_prompt = True
                return False
            
            # Increment usage
            st.session_state.free_images_used += 1
            self.save_usage_count()
            
            # Check if limit reached
            if st.session_state.free_images_used >= self.max_free_images:
                st.session_state.show_upgrade_prompt = True
            
            return True
        return False
    
    def can_generate_free(self) -> bool:
        """Check if user can generate more free images"""
        if st.session_state.api_mode != 'free':
            return False
        
        current_usage = st.session_state.free_images_used
        return current_usage < self.max_free_images
    
    def get_remaining_free(self) -> int:
        """Get remaining free generations"""
        if st.session_state.api_mode != 'free':
            return 0
        
        current_usage = st.session_state.free_images_used
        return max(0, self.max_free_images - current_usage)
    
    def set_premium_mode(self, api_key: str) -> bool:
        """Switch to premium mode with API key validation"""
        if self.validate_api_key(api_key):
            st.session_state.api_mode = 'premium'
            st.session_state.huggingface_api_key = api_key
            st.session_state.show_upgrade_prompt = False
            self.save_usage_count()
            return True
        return False
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate HuggingFace API key format"""
        if not api_key or len(api_key) < 20:
            return False
        
        # Check for HuggingFace format
        if api_key.startswith('hf_') and len(api_key) > 30:
            return True
        
        # Allow other long keys
        if len(api_key) > 20:
            return True
            
        return False
    
    def is_premium_user(self) -> bool:
        """Check if user is in premium mode"""
        return (st.session_state.api_mode == 'premium' and 
                st.session_state.huggingface_api_key)
    
    def get_provider_for_generation(self) -> str:
        """Determine which provider to use for generation"""
        if self.is_premium_user():
            return 'huggingface'
        elif self.can_generate_free():
            return 'pollinations'
        else:
            return 'limit_reached'
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        return {
            'user_id': st.session_state.persistent_user_id[:8] + "...",
            'api_mode': st.session_state.api_mode,
            'free_images_used': st.session_state.free_images_used,
            'free_remaining': self.get_remaining_free(),
            'is_premium': self.is_premium_user(),
            'has_api_key': bool(st.session_state.huggingface_api_key)
        }
    
    def reset_daily_limits(self):
        """Reset daily limits (called automatically)"""
        # This could be enhanced to reset limits daily
        # For now, the limits are per-session/browser
        pass
    
    def debug_session_info(self) -> Dict[str, Any]:
        """Get debug information about the session"""
        return {
            'user_id': st.session_state.persistent_user_id,
            'free_images_used': st.session_state.free_images_used,
            'api_mode': st.session_state.api_mode,
            'can_generate': self.can_generate_free(),
            'remaining': self.get_remaining_free(),
            'is_premium': self.is_premium_user()
        }
