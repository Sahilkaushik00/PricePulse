import streamlit as st
import os
import json
import re
from typing import TypedDict, List, Annotated, Sequence
import operator
from dotenv import load_dotenv

# Search & LLM imports
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="PricePulse Agent | Real-time Comparison",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar / Settings ---
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter Gemini key...", key="gemini_key")
st.sidebar.text_input("Tavily API Key", type="password", placeholder="Enter Tavily key...", key="tavily_key")
st.sidebar.info("Get your free search API key at [tavily.com](https://tavily.com)")

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700&f[]=plus-jakarta-sans@400,500,600,700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #FBFBF9; }
    h1, h2, h3 { font-family: 'Clash Display', sans-serif; color: #0F172A; }
    .main-title { font-size: 3.5rem; font-weight: 700; margin-bottom: 1rem; line-height: 1.1; }
    .subtitle { color: #64748B; font-size: 1.25rem; margin-bottom: 3rem; }
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
    api_key = st.session_state.get("gemini_key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in the sidebar or environment variables.")
    # Note: Updated to 'gemini-1.5-flash' as 3.8 does not exist yet!
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

def extractor_node(state: AgentState):
    """Extracts items from the input (text or image)."""
    try:
        model = get_model()
    except Exception as e:
        return {"items": [], "error": str(e)}
        
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
            
        items = json.loads(match.group())
        return {"items": items}
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return {"items": [], "error": str(e)}

def comparison_node(state: AgentState):
    """
    Searches the live web using Tavily for real prices, then asks Gemini 
    to extract and compare the totals across platforms.
    """
    items = state.get("items", [])
    location = state.get("location", "USA")

    if not items:
        return state
        
    try:
        model = get_model()
        tavily_key = st.session_state.get("tavily_key") or os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError("Tavily API Key is missing. Please add it in the sidebar.")
        tavily = TavilyClient(api_key=tavily_key)
    except Exception as e:
        return {"comparison": [], "error": str(e)}

    # Determine candidate platforms based on item categories
    grocery_categories = ["grocery", "fruit", "vegetable", "fresh", "food"]
    is_grocery_list = all(
        any(category in str(item.get("category", "")).lower() or category in str(item.get("name", "")).lower() for category in grocery_categories)
        for item in items
    )
    platforms = ["Instacart", "DoorDash", "Zepto", "Blinkit", "Swiggy Instamart"] if is_grocery_list else ["Amazon", "Walmart", "Target", "Flipkart", "BigBasket"]

    # --- NEW: LIVE WEB SEARCH STEP ---
    st.write(f"🌐 *Agent Step: Searching live web for current prices in **{location}**...*")
    search_contexts = []
    
    for item in items:
        query = f"buy {item.get('quantity', '1')} {item.get('name', '')} price online {location} {' '.join(platforms)}"
        try:
            # max_results=3 keeps the LLM context window clean and focused
            search_res = tavily.search(query=query, search_depth="basic", max_results=3)
            context_str = f"\n--- Real Web Search Results for '{item.get('name')}' ---\n"
            for res in search_res.get("results", []):
                context_str += f"- Source: {res['url']}\n  Snippet: {res['content']}\n"
            search_contexts.append(context_str)
        except Exception as e:
            st.warning(f"Tavily search failed for {item.get('name')}: {str(e)}")
            
    aggregated_context = "".join(search_contexts)

    # --- LLM COMPARISON STEP ---
    st.write("🤖 *Agent Step: Analyzing live search results to find the lowest complete basket...*")
    
    items_payload = json.dumps([{"name": i.get("name", ""), "quantity": i.get("quantity", ""), "category": i.get("category", "")} for i in items], ensure_ascii=False)

    prompt = f"""
You are a shopping comparison agent.

Location: {location}
Requested shopping list:
{items_payload}

Candidate platforms: {", ".join(platforms)}

======================
LIVE WEB SEARCH RESULTS (TAVILY):
{aggregated_context}
======================

TASK:
Based PRIMARILY on the live web search results above, find the current price of EVERY requested item on EVERY candidate platform where the item is available. 

IMPORTANT RULES:
- Extract prices and real product URLs from the search context where available. 
- If a specific platform is missing from the search results, you may estimate the typical local price to ensure the comparison completes, but prioritize the real data.
- "price" must be the TOTAL line-item price for the requested quantity.
- Do not mix platforms. A platform is eligible only if ALL requested items are available on it.
- If an item is totally unavailable on a platform, use available=false and price=null.

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
          "link": "Direct URL from search results if possible"
        }}
      ]
    }}
  ]
}}
"""

    try:
        response = model.invoke(prompt)
        res_text = response.content
        if isinstance(res_text, list):
            res_text = "".join(p if isinstance(p, str) else p.get("text", "") if isinstance(p, dict) else "" for p in res_text)

        match = re.search(r"\{.*\}", res_text, re.DOTALL)
        if not match: raise ValueError("No JSON object found in model response.")

        data = json.loads(match.group())
        offers = data.get("offers", [])
        requested_items = {str(i.get("name", "")).strip().lower(): i for i in items}
        platform_summaries = []

        for platform_data in offers:
            platform_name = str(platform_data.get("platform", "")).strip()
            currency = str(platform_data.get("currency", "") or "USD").strip()
            if not platform_name or not isinstance(platform_data.get("items", []), list): continue

            offers_by_item = {str(o.get("itemName", "")).strip().lower(): o for o in platform_data.get("items", []) if isinstance(o, dict)}
            
            line_items, total, complete = [], 0.0, True
            for requested_name, requested_item in requested_items.items():
                offer = offers_by_item.get(requested_name)
                if not offer or not offer.get("available", False):
                    complete = False; break
                try:
                    line_price = float(offer.get("price"))
                    if line_price < 0: raise ValueError
                except:
                    complete = False; break
                    
                total += line_price
                line_items.append({
                    "itemName": offer.get("itemName", requested_item.get("name", "")),
                    "quantity": offer.get("quantity", requested_item.get("quantity", "")),
                    "platform": platform_name,
                    "price": line_price,
                    "currency": currency,
                    "link": offer.get("link", "#"),
                })

            if complete and len(line_items) == len(requested_items):
                platform_summaries.append({
                    "platform": platform_name,
                    "currency": currency,
                    "total": total,
                    "itemCount": len(line_items),
                    "items": line_items,
                })

        if not platform_summaries:
            raise ValueError("Based on the live data, no single platform could fulfill the complete list.")

        platform_summaries.sort(key=lambda x: (x["total"], x["platform"].lower()))
        winner = platform_summaries[0]

        return {
            "comparison": winner["items"],
            "platform_totals": [{"platform": s["platform"], "currency": s["currency"], "total": s["total"], "itemCount": s["itemCount"]} for s in platform_summaries],
            "recommended_platform": winner["platform"],
        }
    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")
        return {"comparison": [], "platform_totals": [], "recommended_platform": "", "error": str(e)}

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
st.markdown('<p class="subtitle">Live web-searched comparison across top platforms to find your lowest complete basket.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 Your List")
    user_location = st.text_input("Your Location:", placeholder="e.g. San Francisco, CA or Mumbai, India")
    input_type = st.radio("Choose input method:", ["Text Input", "Upload Photo"], horizontal=True)
    
    user_input = ""
    image_input = None
    if input_type == "Text Input":
        user_input = st.text_area("List your products:", placeholder="e.g. 2kg apples, 1 pack of eggs...", height=150)
    else:
        image_input = st.file_uploader("Upload a photo of your list:", type=["jpg", "jpeg", "png"])
        if image_input: st.image(image_input, use_column_width=True)

    if st.button("Find Lowest Total", type="primary", use_container_width=True):
        if not user_input and not image_input:
            st.warning("Please provide some input.")
        elif not user_location:
            st.warning("Please provide your location for localized results.")
        else:
            with st.status("Agent is working...", expanded=True):
                content = user_input if user_input else "Analyze the attached image"
                inputs = {
                    "messages": [HumanMessage(content=content)], 
                    "items": [], "comparison": [], "platform_totals": [], 
                    "recommended_platform": "", "location": user_location, "error": ""
                }
                
                try:
                    result = app.invoke(inputs)
                    if result.get("error"):
                        st.error(f"Pipeline Halted: {result['error']}")
                    else:
                        st.session_state.results = result.get("comparison", [])
                        st.session_state.items = result.get("items", [])
                        st.session_state.platform_totals = result.get("platform_totals", [])
                        st.session_state.recommended_platform = result.get("recommended_platform", "")
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

        if recommended_platform and platform_totals:
            winner = next((p for p in platform_totals if p.get("platform") == recommended_platform), None)
            if winner:
                st.success(f"🏆 Best complete basket: **{winner['platform']}** — **{winner['currency']} {winner['total']:.2f}**")

        if platform_totals:
            st.markdown("#### Complete basket comparison")
            st.dataframe([{"Platform": p.get("platform", "N/A"), "Items": p.get("itemCount", 0), "Total": (f"{p.get('currency', 'USD')} {float(p.get('total', 0)):.2f}")} for p in platform_totals], use_container_width=True, hide_index=True)

        st.markdown("#### Recommended basket")
        st.table(results_data)

        total = sum(float(r.get("price", 0) or 0) for r in results_data)
        currency = results_data[0].get("currency", "USD") if results_data else "USD"

        st.divider()
        st.metric(label="Lowest Complete Basket Total", value=f"{currency} {total:.2f}")

        PLATFORM_HOME_URLS = {
            "instacart": "https://www.instacart.com/", "doordash": "https://www.doordash.com/",
            "zepto": "https://www.zeptonow.com/", "blinkit": "https://blinkit.com/",
            "swiggy instamart": "https://www.swiggy.com/instamart", "amazon": "https://www.amazon.com/",
            "walmart": "https://www.walmart.com/", "target": "https://www.target.com/",
            "flipkart": "https://www.flipkart.com/", "bigbasket": "https://www.bigbasket.com/",
        }

        def get_platform_url(platform_name, basket_rows):
            for row in basket_rows:
                link = row.get("link")
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    return link
            return PLATFORM_HOME_URLS.get(str(platform_name or "").strip().lower(), "#")

        shopping_url = get_platform_url(recommended_platform, results_data)

        if recommended_platform and shopping_url != "#":
            st.link_button(f"🛒 Shop on {recommended_platform} — Lowest Total", shopping_url, use_container_width=True)
            st.caption("The basket is optimized for one platform. The button opens the winning platform; you may need to add the listed items to its cart manually.")
        else:
            st.warning("No valid platform link was returned for the recommended basket.")
