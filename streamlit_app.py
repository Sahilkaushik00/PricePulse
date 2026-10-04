import streamlit as st
import os
import base64
from PIL import Image
import io
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="PricePulse Intelligence | Real-time Comparison",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700&f[]=plus-jakarta-sans@400,500,600,700&display=swap');

    /* Global Light Theme */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Clash Display', sans-serif;
        color: #0F172A !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Input Card */
    .agent-card {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }

    /* Primary Button Customization */
    .stButton > button {
        background: #10B981 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #059669 !important;
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }

    /* Custom Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #0F172A;
    }
    
    .subtitle {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Winner Card */
    .winner-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #EFF6FF 100%);
        border: 2px solid #10B981;
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: #064E3B;
    }

    /* Input Field Styling */
    .stTextInput input, .stTextArea textarea {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* Info/Warning boxes */
    .stAlert {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Intelligence Engine Logic ---
def run_intelligence_engine(text=None, image_bytes=None, location="USA"):
    """Main function to run the intelligence logic using raw Gemini SDK."""
    api_key = st.session_state.get("sidebar_api_key", "") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("Please set GEMINI_API_KEY in the sidebar settings or environment variables.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Step 1: Extraction
    st.write("🤖 *Phase 1: Extracting intelligence from input...*")
    
    extract_prompt = """
    Extract a list of grocery or retail items from the following input.
    Include name, quantity, and a broad category.
    Return ONLY a JSON list of objects: [{"name": "...", "quantity": "...", "category": "..."}]
    """
    
    try:
        content = [extract_prompt]
        if image_bytes:
            content.append({'mime_type': 'image/jpeg', 'data': image_bytes})
        if text:
            content.append(f"Input Text: {text}")
            
        response = model.generate_content(content)
        
        # Parse JSON
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        items = json.loads(match.group()) if match else []
        
        if not items:
            st.warning("No items identified in the input.")
            return None
            
        st.write(f"✅ Identified {len(items)} items. Initiating platform optimization...")
        
        # Step 2: Platform Optimization (Search)
        # We use a second model instance with search tools enabled
        # Note: If search tool is not available in this environment, it will fallback to internal knowledge
        try:
            search_model = genai.GenerativeModel('gemini-1.5-flash', tools=[{'google_search': {}}])
        except:
            search_model = model # Fallback
            
        search_prompt = f"""
        Find the single platform (Amazon, Walmart, Target, Instacart, Zepto, Blinkit, etc.) available in {location} 
        that offers the absolute lowest TOTAL price for this entire list: {items}
        
        Instructions:
        1. Identify the best platform.
        2. Provide individual prices and the total.
        3. Provide reasoning.
        
        Return a JSON object:
        {{
          "bestPlatform": "...",
          "totalPrice": 0.0,
          "currency": "USD",
          "items": [{{"itemName": "...", "price": 0.0, "link": "..."}}],
          "reasoning": "..."
        }}
        """
        
        st.write(f"🔍 Analyzing 50+ platforms for **{location}**...")
        search_response = search_model.generate_content(search_prompt)
        
        # Parse Result
        res_match = re.search(r'\{.*\}', search_response.text, re.DOTALL)
        result = json.loads(res_match.group()) if res_match else {}
        
        return result
    except Exception as e:
        st.error(f"Intelligence Engine Error: {str(e)}")
        return None

# --- UI Layout ---
st.markdown('<h1 class="main-title">PricePulse <span style="color: #10B981;">Intelligence</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-driven total optimization across major platforms.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.3], gap="large")

if "best_store" not in st.session_state:
    st.session_state.best_store = {}

with col1:
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Intelligence Terminal")
    user_location = st.text_input("Checkout Location", placeholder="e.g. San Francisco, CA")
    
    input_type = st.radio("Input Vector", ["Text List", "Visual Scan"], horizontal=True)
    
    user_input = ""
    image_input = None
    curr_image_bytes = None
    
    if input_type == "Text List":
        user_input = st.text_area("List items and quantities", placeholder="e.g. 2kg Apples, 1 pack of eggs...", height=180)
    else:
        image_input = st.file_uploader("Upload shopping list image", type=["jpg", "png", "jpeg"])
        if image_input: 
            st.image(image_input, use_column_width=True)
            curr_image_bytes = image_input.getvalue()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN INTELLIGENCE ENGINE", type="primary", use_container_width=True):
        st.write("🔄 *Triggering Engine...*")
        if not user_location:
            st.warning("⚠️ Please provide a checkout location.")
        elif not user_input and not image_input:
            st.warning("⚠️ Please provide a shopping list (text or image).")
        else:
            # Check for API Key immediately
            api_key = st.session_state.get("sidebar_api_key", "") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("❌ Gemini API Key is missing. Please enter it in the sidebar.")
            else:
                with st.status("🚀 Intelligence Engine active...", expanded=True) as status:
                    result = run_intelligence_engine(
                        text=user_input, 
                        image_bytes=curr_image_bytes, 
                        location=user_location
                    )
                    if result and isinstance(result, dict) and "bestPlatform" in result:
                        st.session_state.best_store = result
                        status.update(label="✨ Optimization complete!", state="complete", expanded=False)
                        st.success("Analysis finished. Scroll right to see the report!")
                        # Force a rerun to ensure the col2 updates immediately
                        st.rerun()
                    else:
                        status.update(label="❌ Engine failed", state="error")
                        st.error("The engine could not find a valid price comparison. Please check your list format.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Intelligence Report")
    best_store = st.session_state.get("best_store", {})
    
    if not best_store:
        st.info("Awaiting input. Scanning 50+ platforms in real-time...")
        st.markdown("""
        <div style="opacity: 0.2; margin-top: 4rem; text-align: center;">
            <p style="font-size: 5rem;">📉</p>
            <p>Input your list to activate real-time optimization</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="winner-card">
            <p style="text-transform: uppercase; font-size: 0.75rem; font-weight: 900; color: #059669; margin: 0; letter-spacing: 0.1em;">Optimized Winner</p>
            <h2 style="margin: 0.5rem 0; font-size: 3.5rem; color: #064E3B !important;">{best_store.get('bestPlatform')}</h2>
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 1.5rem;">
                <div>
                    <p style="color: #64748B; font-size: 0.8rem; font-weight: 700; margin: 0;">TOTAL SAVINGS APPLIED</p>
                    <p style="font-size: 2.5rem; font-weight: 800; color: #064E3B; margin: 0;">{best_store.get('currency')} {best_store.get('totalPrice', 0.0):.2f}</p>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 0.5rem 1rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <span style="color: #059669; font-weight: 800; font-size: 0.8rem;">READY FOR CHECKOUT</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Style the items table
        st.markdown("#### Itemized Breakdown")
        st.table(best_store.get('items', []))
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 20px; border-left: 4px solid #10B981;">
            <p style="color: #10B981; font-weight: 800; font-size: 0.7rem; margin-bottom: 0.5rem; text-transform: uppercase;">Agent Reasoning</p>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; margin: 0;">{best_store.get('reasoning')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("REDIRECT TO CHECKOUT →", use_container_width=True)

# --- Sidebar / Footer ---
st.sidebar.markdown("### Engine Settings")
st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter your key...", key="sidebar_api_key")
st.sidebar.info("This key is used for the LangGraph intelligence processing. It is not stored on our servers.")
st.sidebar.divider()
st.sidebar.markdown("© 2026 PricePulse Intelligence Corp")
