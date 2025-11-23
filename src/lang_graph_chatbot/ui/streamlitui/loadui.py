import streamlit as st
from src.lang_graph_chatbot.ui.uiconfigfile import Config

class LoadStreamlitUi:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_Streamlit_Ui(self):
        page_title = self.config.get_PAGE_TITLE()
        if not page_title:
            page_title = "Financial Advisor ChatBot"
        
        st.set_page_config(page_title="💰 " + page_title, layout='wide')
        
        # Custom header
        st.markdown("""
            <h1 style='text-align: center; color: #1f77b4;'>
                💰 AI Financial Advisor
            </h1>
            <p style='text-align: center; color: #666;'>
                Powered by LangGraph • Gemini 🔷 • HuggingFace 🤗
            </p>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("## ⚙️ Settings")
            
            # LLM Provider Selection (NEW!)
            st.markdown("### 🤖 AI Provider")
            provider_choice = st.radio(
                "Choose your AI provider:",
                ["🔷 Google Gemini (Fast)", "🤗 HuggingFace (Unlimited)"],
                index=0,
                help="Gemini: Faster responses, 1500/day limit\nHuggingFace: Unlimited, slower"
            )
            
            # Set provider based on choice
            if "Gemini" in provider_choice:
                self.user_controls["Selected_llm"] = "gemini"
                
                # Gemini model selection
                gemini_models = self.config.get_Gemini_MODEL_OPTIONS()
                self.user_controls["Selected_model"] = st.selectbox(
                    "📦 Select Model",
                    gemini_models,
                    index=0,
                    help="gemini-1.5-flash: Fastest, FREE\ngemini-1.5-pro: Best quality, FREE"
                )
                
                # Gemini API Key input
                self.user_controls["gemini_api_key"] = st.text_input(
                    "🔑 Gemini API Key",
                    type="password",
                    help="Get FREE key at https://aistudio.google.com/app/apikey"
                )
                
                # Token validation
                if self.user_controls["gemini_api_key"]:
                    if len(self.user_controls["gemini_api_key"]) > 30:
                        st.success("✅ API key format looks good!")
                    else:
                        st.warning("⚠️ API key seems too short")
                
                # Provider info
                st.info("🔷 **Gemini Free Tier**\n\n• 60 requests/min\n• 1500 requests/day\n• Fast responses ⚡")
                
            else:  # HuggingFace
                self.user_controls["Selected_llm"] = "huggingface"
                
                # HuggingFace model selection (only working models)
                hf_models = self.config.get_HuggingFace_MODEL_OPTIONS()
                self.user_controls["Selected_model"] = st.selectbox(
                    "📦 Select Model",
                    hf_models,
                    index=0,
                    help="Mistral & Zephyr are most reliable"
                )
                
                # HuggingFace API Token input
                self.user_controls["huggingface_api_token"] = st.text_input(
                    "🔑 HuggingFace API Token",
                    type="password",
                    help="Get FREE token at https://huggingface.co/settings/tokens"
                )
                
                # Token validation
                if self.user_controls["huggingface_api_token"]:
                    if self.user_controls["huggingface_api_token"].startswith("hf_"):
                        st.success("✅ Token format looks good!")
                    else:
                        st.warning("⚠️ Token should start with 'hf_'")
                
                # Provider info
                st.success("🤗 **HuggingFace Free Tier**\n\n• Unlimited requests\n• No credit card\n• 100% FREE forever")
            
            # Quick setup guides
            with st.expander("🚀 Quick Setup Guide"):
                if self.user_controls["Selected_llm"] == "gemini":
                    st.markdown("""
                    **Get FREE Gemini API Key (30 sec):**
                    
                    1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
                    2. Click **"Get API Key"**
                    3. Click **"Create API key"**
                    4. Copy and paste above ☝️
                    
                    **No credit card required!** ✅
                    """)
                else:
                    st.markdown("""
                    **Get FREE HuggingFace Token (30 sec):**
                    
                    1. Go to [HuggingFace](https://huggingface.co/join)
                    2. Sign up (FREE)
                    3. Go to [Settings → Tokens](https://huggingface.co/settings/tokens)
                    4. Click **"New token"**
                    5. Select **"Read"** access
                    6. Copy and paste above ☝️
                    """)
            
            st.markdown("---")
            
            # Use case selection
            user_option = self.config.get_Usecase_Options()
            self.user_controls["selected_user_option"] = st.selectbox(
                "📋 Select Use Case",
                user_option,
                help="Choose the type of financial assistance you need"
            )
            
            st.markdown("---")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                if "messages" in st.session_state:
                    st.session_state["messages"] = []
                st.rerun()
            
            st.markdown("---")
            
            # Features info
            with st.expander("ℹ️ What Can I Do?"):
                st.markdown("""
                **Financial Services:**
                - 🏆 Gold purchase planning
                - 💰 Budget analysis & optimization
                - 📊 Investment recommendations
                - 💵 Savings strategies
                - 📈 Financial goal planning
                
                **Example Questions:**
                ```
                "I want to buy gold, salary 50k, expenses 30k"
                
                "Analyze my budget: income 80k, expenses 60k"
                
                "Should I invest in gold or mutual funds?"
                
                "Help me save 20k per month"
                ```
                """)
            
            # Provider comparison
            with st.expander("⚖️ Provider Comparison"):
                st.markdown("""
                | Feature | Gemini | HuggingFace |
                |---------|--------|-------------|
                | Speed | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
                | Limit | 1500/day | Unlimited |
                | Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
                | Setup | 30 seconds | 30 seconds |
                | Cost | FREE | FREE |
                
                **Recommendation:** Use Gemini for best experience!
                """)
            
            st.markdown("---")
            
            # Branding
            st.markdown("""
                <div style='text-align: center; color: #666; font-size: 12px;'>
                    <p>Built with ❤️ using</p>
                    <p>LangGraph • Gemini 🔷 • HuggingFace 🤗</p>
                    <p style='margin-top: 10px; font-size: 10px;'>
                        100% Free & Open Source
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        return self.user_controls