import streamlit as st
import os
import base64
from PIL import Image
import io
import json
import re
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
    platform_totals: List[dict]
    recommended_platform: str
    location: str
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
    
    last_msg = state["messages"][-1]
    
    st.write("🤖 *Agent Step: Extracting items from input...*")
    
    # We use the model to extract structured data
    # Note: In a real app, you'd use LangChain's with_structured_output
    # For simplicity, we'll use a direct prompt and parse JSON
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
        
        import json
        import re
        # Basic JSON extraction from markdown
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
    """
    Compare the COMPLETE shopping list on each platform and recommend
    the single platform with the lowest basket total.

    This intentionally does not select the cheapest platform independently
    for each item, because that would create a mixed-platform basket.
    """
    items = state.get("items", [])
    location = state.get("location", "USA")

    if not items:
        return state

    st.write(
        f"🤖 *Agent Step: Comparing complete basket totals across platforms for **{location}**...*"
    )

    model = get_model()
    if not model:
        return state

    # Use a sensible set of candidate platforms. The model should return
    # an offer for every requested item on every platform where possible.
    grocery_categories = ["grocery", "fruit", "vegetable", "fresh", "food"]
    is_grocery_list = all(
        any(
            category in str(item.get("category", "")).lower()
            or category in str(item.get("name", "")).lower()
            for category in grocery_categories
        )
        for item in items
    )

    platforms = (
        ["Instacart", "DoorDash", "Zepto", "Blinkit", "Swiggy Instamart"]
        if is_grocery_list
        else ["Amazon", "Walmart", "Target", "Flipkart", "BigBasket"]
    )

    items_payload = json.dumps(
        [
            {
                "name": item.get("name", ""),
                "quantity": item.get("quantity", ""),
                "category": item.get("category", ""),
            }
            for item in items
        ],
        ensure_ascii=False,
    )

    prompt = f"""
You are a shopping comparison agent.

Location: {location}

Requested shopping list:
{items_payload}

Candidate platforms:
{", ".join(platforms)}

TASK:
Find the current price of EVERY requested item on EVERY candidate platform
where the item is available.

IMPORTANT RULES:
- Use the exact requested quantity for each item.
- "price" must be the TOTAL line-item price for that requested quantity,
  not a unit price.
- Do not mix platforms when deciding the winner.
- A platform is eligible only if ALL requested items are available on that
  same platform.
- Keep product size/pack count consistent with the requested quantity.
- Use the same currency for all offers based on the location.
- Include a direct product URL when available.
- If an item is unavailable on a platform, use available=false and price=null.

Return ONLY valid JSON in exactly this structure:
{{
  "offers": [
    {{
      "platform": "Platform Name",
      "currency": "USD",
      "items": [
        {{
          "itemName": "Requested item name",
          "quantity": "Requested quantity",
          "price": 0.0,
          "available": true,
          "link": "Direct URL"
        }}
      ]
    }}
  ]
}}

Each platform should have an entry for every requested item.
Do not return a pre-calculated mixed-platform total; the application will
calculate the complete basket totals in Python.
"""

    try:
        response = model.invoke(prompt)
        res_text = response.content

        if isinstance(res_text, list):
            res_text = "".join(
                part if isinstance(part, str)
                else part.get("text", "") if isinstance(part, dict)
                else ""
                for part in res_text
            )

        match = re.search(r"\{.*\}", res_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model response.")

        data = json.loads(match.group())
        offers = data.get("offers", [])

        if not isinstance(offers, list):
            raise ValueError("Invalid response: 'offers' must be a list.")

        # Normalize requested names so matching is case-insensitive.
        requested_items = {
            str(item.get("name", "")).strip().lower(): item
            for item in items
        }

        platform_summaries = []

        for platform_data in offers:
            platform_name = str(
                platform_data.get("platform", "")
            ).strip()
            currency = str(
                platform_data.get("currency", "") or "USD"
            ).strip()

            if not platform_name:
                continue

            raw_items = platform_data.get("items", [])
            if not isinstance(raw_items, list):
                continue

            offers_by_item = {
                str(offer.get("itemName", "")).strip().lower(): offer
                for offer in raw_items
                if isinstance(offer, dict)
            }

            line_items = []
            total = 0.0
            complete = True

            for requested_name, requested_item in requested_items.items():
                offer = offers_by_item.get(requested_name)

                if not offer or not offer.get("available", False):
                    complete = False
                    break

                try:
                    line_price = float(offer.get("price"))
                except (TypeError, ValueError):
                    complete = False
                    break

                if line_price < 0:
                    complete = False
                    break

                total += line_price

                line_items.append(
                    {
                        "itemName": offer.get(
                            "itemName", requested_item.get("name", "")
                        ),
                        "quantity": offer.get(
                            "quantity", requested_item.get("quantity", "")
                        ),
                        "platform": platform_name,
                        "price": line_price,
                        "currency": currency,
                        "link": offer.get("link", "#"),
                    }
                )

            # Only compare platforms that can fulfill the COMPLETE list.
            if complete and len(line_items) == len(requested_items):
                platform_summaries.append(
                    {
                        "platform": platform_name,
                        "currency": currency,
                        "total": total,
                        "itemCount": len(line_items),
                        "items": line_items,
                    }
                )

        if not platform_summaries:
            raise ValueError(
                "No single platform can fulfill the complete shopping list."
            )

        # THE KEY LOGIC:
        # Sort by COMPLETE basket total, not individual item price.
        platform_summaries.sort(
            key=lambda x: (x["total"], x["platform"].lower())
        )
        winner = platform_summaries[0]

        st.write(
            f"✅ *Lowest complete basket: **{winner['platform']}** — "
            f"{winner['currency']} {winner['total']:.2f}*"
        )

        return {
            # Only show the winning platform's item rows as the recommended basket.
            "comparison": winner["items"],
            "platform_totals": [
                {
                    "platform": summary["platform"],
                    "currency": summary["currency"],
                    "total": summary["total"],
                    "itemCount": summary["itemCount"],
                }
                for summary in platform_summaries
            ],
            "recommended_platform": winner["platform"],
        }

    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")
        return {
            "comparison": [],
            "platform_totals": [],
            "recommended_platform": "",
            "error": str(e),
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
st.markdown('<p class="subtitle">The lowest total for your entire list on a single platform, calculated by AI.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 Your List")
    
    # Location Input
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

    if st.button("Find Lowest Total", type="primary", use_container_width=True):
        if not user_input and not image_input:
            st.warning("Please provide some input.")
        elif not user_location:
            st.warning("Please provide your location for localized results.")
        else:
            with st.status("Agent is working...", expanded=True) as status:
                # Prepare inputs
                content = user_input if user_input else "Analyze the attached image"
                inputs = {
                    "messages": [HumanMessage(content=content)], 
                    "items": [],
                    "comparison": [],
                    "platform_totals": [],
                    "recommended_platform": "",
                    "location": user_location,
                    "error": ""
                }
                
                try:
                    # Run LangGraph Agent
                    result = app.invoke(inputs)
                    st.session_state.results = result.get("comparison", [])
                    st.session_state.items = result.get("items", [])
                    st.session_state.platform_totals = result.get("platform_totals", [])
                    st.session_state.recommended_platform = result.get(
                        "recommended_platform", ""
                    )
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Agent Error: {str(e)}")

with col2:
    st.markdown("### 📊 Optimization Results")
    
    results_data = st.session_state.get("results", [])
    
    if not results_data:
        st.info("Results will appear here after analysis.")
    else:
        platform_totals = st.session_state.get("platform_totals", [])
        recommended_platform = st.session_state.get("recommended_platform", "")

        # Show the winner first.
        if recommended_platform and platform_totals:
            winner = next(
                (
                    p for p in platform_totals
                    if p.get("platform") == recommended_platform
                ),
                None,
            )

            if winner:
                st.success(
                    f"🏆 Best complete basket: **{winner['platform']}** — "
                    f"**{winner['currency']} {winner['total']:.2f}**"
                )

        # Show every eligible platform's COMPLETE basket total so the user
        # can see why the recommended platform won.
        if platform_totals:
            st.markdown("#### Complete basket comparison")
            st.dataframe(
                [
                    {
                        "Platform": p.get("platform", "N/A"),
                        "Items": p.get("itemCount", 0),
                        "Total": (
                            f"{p.get('currency', 'USD')} "
                            f"{float(p.get('total', 0)):.2f}"
                        ),
                    }
                    for p in platform_totals
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Recommended basket")
        st.table(results_data)

        total = sum(
            float(r.get("price", 0) or 0)
            for r in results_data
        )
        currency = (
            results_data[0].get("currency", "USD")
            if results_data else "USD"
        )

        st.divider()
        st.metric(
            label="Lowest Complete Basket Total",
            value=f"{currency} {total:.2f}"
        )
        st.button("Add All to Cart", use_container_width=True)

# --- Sidebar / Footer ---
st.sidebar.markdown("### Settings")
st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter your key...")
st.sidebar.info("This key is used for the LangGraph agent processing.")
