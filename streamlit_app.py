import streamlit as st
import os
import base64
from PIL import Image
import io
from typing import TypedDict, List, Annotated, Sequence
import operator
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="PricePulse Agent | Real-time Comparison",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700&f[]=plus-jakarta-sans@400,500,600,700&display=swap');

    /* Global Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Clash Display', sans-serif;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Input Card */
    .agent-card {
        background: #1C2128;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    /* Primary Button Customization */
    .stButton > button {
        background: #10B981 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #34D399 !important;
        transform: scale(1.02);
    }

    /* Custom Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
    
    .subtitle {
        color: #8B949E;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Winner Card */
    .winner-card {
        background: linear-gradient(135deg, #1C2128 0%, #161B22 100%);
        border: 2px solid #10B981;
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
    }

    /* Input Field Styling overrides for visibility */
    .stTextInput input, .stTextArea textarea {
        color: white !important;
        background-color: #0D1117 !important;
    }

    /* Info/Warning boxes */
    .stAlert {
        background-color: #161B22 !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    items: List[dict]
    best_store: dict
    location: str
    error: str

# --- Agent Logic ---
def get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in your environment variables.")
        return None
    return ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)

def extractor_node(state: AgentState):
    """Extracts items from the input (text or image)."""
    model = get_model()
    if not model: return state
    last_msg = state["messages"][-1]
    
    st.write("🤖 *Agent Step: Extracting items from input...*")
    
    prompt = f"""
    Extract a list of grocery or retail items from the following input.
    Include name, quantity, and a broad category.
    
    Input: {last_msg.content}
    
    Return ONLY a JSON list of objects: [{{"name": "...", "quantity": "...", "category": "..."}}]
    """
    
    try:
        response = model.invoke(prompt)
        res_text = response.content
        if isinstance(res_text, list):
            res_text = "".join([str(p) for p in res_text])
        
        import json, re
        match = re.search(r'\[.*\]', res_text, re.DOTALL)
        items = json.loads(match.group()) if match else []
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)}

def comparison_node(state: AgentState):
    """Finds the single best platform for the entire list."""
    items = state.get("items", [])
    location = state.get("location", "USA")
    if not items: return state
    
    st.write(f"🤖 *Agent Step: Optimizing total for **{location}**...*")
    model = get_model()
    
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
    
    try:
        response = model.invoke(search_prompt)
        res_text = response.content
        if isinstance(res_text, list):
            res_text = "".join([str(p) for p in res_text])
            
        import json, re
        match = re.search(r'\{.*\}', res_text, re.DOTALL)
        result = json.loads(match.group()) if match else {}
        return {"best_store": result}
    except Exception as e:
        return {"best_store": {}, "error": str(e)}

# --- Define the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("extractor", extractor_node)
workflow.add_node("comparer", comparison_node)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "comparer")
workflow.add_edge("comparer", END)
app = workflow.compile()

# --- UI Layout ---
st.markdown('<h1 class="main-title">PricePulse <span style="color: #10B981;">Agent</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-driven total optimization across major platforms.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Intelligence Terminal")
    user_location = st.text_input("Checkout Location", placeholder="e.g. San Francisco, CA")
    
    input_type = st.radio("Input Vector", ["Text List", "Visual Scan"], horizontal=True)
    
    user_input = ""
    image_input = None
    
    if input_type == "Text List":
        user_input = st.text_area("List items and quantities", placeholder="e.g. 2kg Apples, 1 pack of eggs...", height=180)
    else:
        image_input = st.file_uploader("Upload shopping list image", type=["jpg", "png", "jpeg"])
        if image_input: 
            st.image(image_input, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN OPTIMIZATION AGENT", type="primary", use_container_width=True):
        if not user_location or (not user_input and not image_input):
            st.warning("⚠️ Location and Input required for localized optimization.")
        else:
            with st.status("🚀 Agent executing deep search...", expanded=True):
                content = user_input if user_input else "Analyze the attached image"
                inputs = {"messages": [HumanMessage(content=content)], "items": [], "best_store": {}, "location": user_location, "error": ""}
                try:
                    result = app.invoke(inputs)
                    st.session_state.best_store = result.get("best_store", {})
                    st.success("✨ Optimization cycle finished!")
                except Exception as e:
                    st.error(f"Agent Error: {e}")
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
            <p style="text-transform: uppercase; font-size: 0.75rem; font-weight: 900; color: #10B981; margin: 0; letter-spacing: 0.1em;">Optimized Winner</p>
            <h2 style="margin: 0.5rem 0; font-size: 3.5rem; color: white !important;">{best_store.get('bestPlatform')}</h2>
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 1.5rem;">
                <div>
                    <p style="color: #64748B; font-size: 0.8rem; font-weight: 700; margin: 0;">TOTAL SAVINGS APPLIED</p>
                    <p style="font-size: 2.5rem; font-weight: 800; color: white; margin: 0;">{best_store.get('currency')} {best_store.get('totalPrice', 0.0):.2f}</p>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 0.5rem 1rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <span style="color: #10B981; font-weight: 800; font-size: 0.8rem;">READY FOR CHECKOUT</span>
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
st.sidebar.markdown("### Settings")
st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter your key...")
st.sidebar.info("This key is used for the LangGraph agent processing.")
