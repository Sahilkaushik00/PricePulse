import streamlit as st
import os
import base64
from PIL import Image
import io
from typing import TypedDict, List, Annotated, Sequence
import operator
import json
import re
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="PricePulse Agent | Real-time Comparison",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700&f[]=plus-jakarta-sans@400,500,600,700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #FBFBF9;
    }
    
    h1, h2, h3 {
        font-family: 'Clash Display', sans-serif;
        color: #0F172A;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .subtitle {
        color: #64748B;
        font-size: 1.25rem;
        margin-bottom: 3rem;
    }
    
    .agent-card {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid #F1F5F9;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
    }
    
    .price-badge {
        background: #F1F5F9;
        color: #0F172A;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .platform-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    items: List[dict]
    best_platform: dict
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
    Include name, quantity, and a broad category (Grocery, Electronics, Apparel, etc.).
    
    Input: {last_msg.content}
    
    Return ONLY a JSON list of objects: [{{"name": "...", "quantity": "...", "category": "..."}}]
    """
    
    try:
        response = model.invoke(prompt)
        res_text = response.content
        if isinstance(res_text, list):
            res_text = "".join([part if isinstance(part, str) else (part.get("text", "") if isinstance(part, dict) else "") for part in res_text])
        
        match = re.search(r'\[.*\]', res_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON list found in response: {res_text}")
        json_str = match.group()
        items = json.loads(json_str)
        return {"items": items}
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return {"items": [], "error": str(e)}

def comparison_node(state: AgentState):
    """Calculates total basket cost per platform and finds the single platform with the lowest total."""
    items = state.get("items", [])
    location = state.get("location", "USA")
    if not items: return state
    
    st.write(f"🤖 *Agent Step: Finding the platform with the lowest total price in **{location}**...*")
    model = get_model()
    
    items_summary = ", ".join([f"{item['quantity']} of {item['name']}" for item in items])
    
    search_prompt = f"""
    You are given a shopping list with the following items: {items_summary}.
    Target location: {location}.
    
    Compare the combined total price for ALL items across popular platforms operating in {location} (e.g., Instacart, Walmart, Amazon, Target, Blinkit, Swiggy Instamart, Zepto, Flipkart).
    
    Determine WHICH SINGLE PLATFORM gives the lowest overall total cost for the entire basket.
    
    Return ONLY a single JSON object formatted exactly as follows:
    {{
        "platform": "Platform Name",
        "total_price": 0.0,
        "currency": "USD",
        "platform_link": "https://www.example.com",
        "item_breakdown": [
            {{"itemName": "Item Name", "quantity": "1", "estimated_price": 0.0}}
        ]
    }}
    """
    
    try:
        response = model.invoke(search_prompt)
        res_text = response.content
        if isinstance(res_text, list):
            res_text = "".join([part if isinstance(part, str) else (part.get("text", "") if isinstance(part, dict) else "") for part in res_text])
        
        match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        json_str = match.group()
        result = json.loads(json_str)
        return {"best_platform": result}
    except Exception as e:
        return {
            "best_platform": {
                "platform": "N/A",
                "total_price": 0.0,
                "currency": "USD",
                "platform_link": "#",
                "item_breakdown": []
            },
            "error": str(e)
        }

# --- Define the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("extractor", extractor_node)
workflow.add_node("comparer", comparison_node)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "comparer")
workflow.add_edge("comparer", END)

app = workflow.compile()

# --- UI Layout ---
st.markdown('<h1 class="main-title">PricePulse <span style="color: #94A3B8;">Agent</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">The lowest total for your entire list, calculated by AI.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📋 Your List")
    
    user_location = st.text_input("Your Location:", placeholder="e.g. San Francisco, CA or Mumbai, India", help="Used to find local quick commerce deals")
    input_type = st.radio("Choose input method:", ["Text Input", "Upload Photo"], horizontal=True)
    
    user_input = ""
    image_input = None
    
    if input_type == "Text Input":
        user_input = st.text_area("List your products:", placeholder="e.g. 2kg apples, 1 pack of eggs...", height=150)
    else:
        image_input = st.file_uploader("Upload a photo of your list:", type=["jpg", "jpeg", "png"])
        if image_input:
            st.image(image_input, use_column_width=True)

    if st.button("Find Lowest Total Platform", type="primary", use_container_width=True):
        if not user_input and not image_input:
            st.warning("Please provide some input.")
        elif not user_location:
            st.warning("Please provide your location for localized results.")
        else:
            with st.status("Agent is working...", expanded=True) as status:
                content = user_input if user_input else "Analyze the attached image"
                inputs = {
                    "messages": [HumanMessage(content=content)], 
                    "items": [], 
                    "best_platform": {}, 
                    "location": user_location,
                    "error": ""
                }
                
                try:
                    result = app.invoke(inputs)
                    st.session_state.best_platform = result.get("best_platform", {})
                    st.session_state.items = result.get("items", [])
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Agent Error: {str(e)}")

with col2:
    st.markdown("### 🏷️ Best Platform Deal")
    
    best_platform = st.session_state.get("best_platform", {})
    
    if not best_platform:
        st.info("Results will appear here after analysis.")
    else:
        platform = best_platform.get("platform", "N/A")
        total_price = best_platform.get("total_price", 0.0)
        currency = best_platform.get("currency", "USD")
        platform_link = best_platform.get("platform_link", "#")
        breakdown = best_platform.get("item_breakdown", [])
        
        # Display Best Platform Card
        st.markdown(f"""
        <div class="agent-card">
            <span class="platform-tag">Lowest Total Cost Platform</span>
            <h2 style="margin-top: 0.5rem; margin-bottom: 0.2rem;">{platform}</h2>
            <p style="color: #64748B; font-size: 0.95rem;">Lowest overall total for all items combined in your list.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if breakdown:
            st.markdown("#### Basket Breakdown")
            st.table(breakdown)
        
        st.divider()
        
        symbol = "₹" if currency.upper() in ["INR", "RS"] else "$"
        st.metric(label=f"Total Cost on {platform}", value=f"{symbol}{total_price:.2f}")
        
        st.write("")
        
        # Add to Cart Button & Platform Link underneath
        st.link_button(f"🛒 Add to Cart on {platform}", url=platform_link, use_container_width=True)
        st.markdown(f"🔗 **Platform Link:** [{platform_link}]({platform_link})")

# --- Sidebar / Footer ---
st.sidebar.markdown("### Settings")
st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter your key...")
st.sidebar.info("This key is used for the LangGraph agent processing.")
