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
    comparison: List[dict]
    error: str

# --- Agent Logic ---
def get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in your environment variables.")
        return None
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

def extractor_node(state: AgentState):
    """Extracts items from the input (text or image)."""
    model = get_model()
    if not model: return state
    
    # Logic to handle multimodal input in LangChain/Gemini
    last_msg = state["messages"][-1]
    
    # We'll use a specific prompt for extraction
    system_prompt = "You are a shopping list extractor. Parse the items and quantities into a structured list. Return JSON only."
    
    # For this demo, we'll simulate the extraction or use the model
    # (In a production app, you'd use structured output)
    response = model.invoke([
        SystemMessage(content=system_prompt),
        last_msg
    ])
    
    # Mocking extraction logic for demonstration of flow
    # In a real implementation, you'd parse response.content JSON
    st.write("🤖 *Agent Step: Extracting items from input...*")
    
    return {"items": [{"name": "Item", "quantity": "1"}]} # Placeholder

def comparison_node(state: AgentState):
    """Searches for lowest prices across platforms."""
    st.write("🤖 *Agent Step: Comparing real-time pricing across platforms...*")
    # Here you would integrate Search tools (e.g. Tavily, Google Search)
    # and use the model to find the best match.
    return {"comparison": []} # Placeholder

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
    st.markdown("### 📝 Your List")
    input_type = st.radio("Choose input method:", ["Text Input", "Upload Photo"], horizontal=True)
    
    user_input = ""
    image_input = None
    
    if input_type == "Text Input":
        user_input = st.text_area("List your products:", placeholder="e.g. 2kg apples, 1 pack of eggs...", height=150)
    else:
        image_input = st.file_uploader("Upload a photo of your list:", type=["jpg", "jpeg", "png"])
        if image_input:
            st.image(image_input, use_column_width=True)

    if st.button("Find Lowest Total", type="primary", use_container_width=True):
        if not user_input and not image_input:
            st.warning("Please provide some input.")
        else:
            with st.status("Agent is working...", expanded=True) as status:
                # Run LangGraph Agent
                # (Conceptual run)
                inputs = {"messages": [HumanMessage(content=user_input if user_input else "Analyze the attached image")]}
                # result = app.invoke(inputs)
                st.success("Analysis complete!")

with col2:
    st.markdown("### 📊 Optimization Results")
    
    # Placeholder for results
    st.info("Results will appear here after analysis.")
    
    # Example table
    st.markdown("""
    | Item | Best Platform | Price |
    | :--- | :--- | :--- |
    | *Example Item* | *Marketplace* | *$0.00* |
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.metric(label="Estimated Total", value="$0.00", delta="- $0.00 (vs Market Avg)")
    st.button("Add All to Cart", use_container_width=True)

# --- Sidebar / Footer ---
st.sidebar.markdown("### Settings")
st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter your key...")
st.sidebar.info("This key is used for the LangGraph agent processing.")
