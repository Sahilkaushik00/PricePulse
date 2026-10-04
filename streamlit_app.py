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
st.sidebar.info("Get your search API key at [tavily.com](https://tavily.com)")

# --- Custom Styling ---
st.markdown("""
<style>
    @It looks like the results or output didn't display on your end. 

Could you let me know what you were looking for or re-state your question? I'll re-run it and get those results right up for you.
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

import json
import re
import os
import streamlit as st
from tavily import TavilyClient

def comparison_node(state: dict) -> dict:
    items = state.get("items", [])
    location = state.get("location", "India")

    if not items:
        return state

    try:
        model = get_model()
        tavily_key = st.session_state.get("tavily_key") or os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError("Tavily API Key missing.")
        tavily = TavilyClient(api_key=tavily_key)
    except Exception as e:
        return {"comparison": [], "error": str(e)}

    # Determine platform set based on product type
    grocery_categories = ["grocery", "fruit", "vegetable", "fresh", "food", "dairy"]
    is_grocery_list = all(
        any(c in str(i.get("category", "")).lower() or c in str(i.get("name", "")).lower() for c in grocery_categories)
        for i in items
    )
    platforms = ["Blinkit", "Zepto", "Swiggy Instamart", "BigBasket"] if is_grocery_list else ["Amazon", "Flipkart", "Walmart"]

    # Step 1: SEARCH INDIVIDUAL ITEMS ACROSS PLATFORMS
    st.write(f"🌐 *Agent Step: Querying live prices item-by-item in **{location}**...*")
    search_contexts = []

    for item in items:
        item_name = item.get("name", "")
        item_qty = item.get("quantity", "")
        
        for platform in platforms:
            query = f'"{item_name}" {item_qty} price on {platform} {location}'
            try:
                search_res = tavily.search(query=query, search_depth="basic", max_results=2)
                for res in search_res.get("results", []):
                    search_contexts.append(
                        f"ITEM_QUERY: {item_name} | QTY: {item_qty}\n"
                        f"PLATFORM: {platform}\n"
                        f"URL: {res.get('url')}\n"
                        f"SNIPPET: {res.get('content')}\n---"
                    )
            except Exception:
                continue

    aggregated_context = "\n".join(search_contexts)

    with st.expander("🔍 Debug: Raw Search Snippets (Item x Platform)"):
        st.text_area("Search Context", value=aggregated_context, height=200)

    # Step 2: EXTRACT ITEM PRICE MATRIX USING LLM
    st.write("🤖 *Agent Step: Parsing prices and building matrix...*")
    items_payload = json.dumps([{"name": i.get("name", ""), "quantity": i.get("quantity", "")} for i in items])

    prompt = f"""
You are an item price extraction model.

Requested Items:
{items_payload}

Search Snippets:
{aggregated_context}

CRITICAL INSTRUCTIONS:
1. Extract explicit price tags found for each requested item on each platform mentioned in snippets.
2. If an item price is not strictly mentioned for a platform, omit it or set "available": false.
3. "price" must be a float representing the total cost for the requested quantity in INR.

Return ONLY valid JSON matching this schema:
{{
  "price_matrix": [
    {{
      "itemName": "Item Name",
      "quantity": "Quantity",
      "platform_prices": [
        {{
          "platform": "Platform Name",
          "price": 50.0,
          "available": true,
          "link": "URL from snippet"
        }}
      ]
    }}
  ]
}}
    try:
        response = model.invoke(prompt)
        res_text = response.content
        if isinstance(res_text, list):
            res_text = "".join(p if isinstance(p, str) else p.get("text", "") if isinstance(p, dict) else "" for p in res_text)

        match = re.search(r"\{.*\}", res_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON matrix returned by LLM.")

        matrix_data = json.loads(match.group()).get("price_matrix", [])

        # Step 3: ANALYZE TOTALS & ITEM-LEVEL WINNERS
        platform_totals = {p: {"platform": p, "total": 0.0, "found_count": 0, "items": []} for p in platforms}
        item_analysis = []
        split_cart_total = 0.0

        for item_entry in matrix_data:
            item_name = item_entry.get("itemName", "")
            quantity = item_entry.get("quantity", "")
            prices = item_entry.get("platform_prices", [])

            valid_offers = [
                p for p in prices 
                if p.get("available") and isinstance(p.get("price"), (int, float)) and p.get("price") > 0
            ]

            if not valid_offers:
                item_analysis.append({
                    "itemName": item_name,
                    "quantity": quantity,
                    "cheapest_platform": "N/A",
                    "best_price": None,
                    "all_offers": []
                })
                continue

            # Sort offers to find cheapest platform for this specific item
            valid_offers.sort(key=lambda x: x["price"])
            best_offer = valid_offers[0]

            split_cart_total += best_offer["price"]
            item_analysis.append({
                "itemName": item_name,
                "quantity": quantity,
                "cheapest_platform": best_offer["platform"],
                "best_price": best_offer["price"],
                "all_offers": valid_offers
            })

            # Accumulate totals per platform
            for offer in valid_offers:
                p_name = offer["platform"]
                if p_name in platform_totals:
                    platform_totals[p_name]["total"] += offer["price"]
                    platform_totals[p_name]["found_count"] += 1
                    platform_totals[p_name]["items"].append({
                        "itemName": item_name,
                        "quantity": quantity,
                        "price": offer["price"],
                        "link": offer.get("link", "#")
                    })

        # Filter out platforms with 0 items found
        active_platforms = [p for p in platform_totals.values() if p["found_count"] > 0]

        if not active_platforms:
            raise ValueError("No explicit item prices could be verified from web search snippets.")

        # Rank single platforms: 1. Most items found, 2. Lowest total cost
        active_platforms.sort(key=lambda x: (-x["found_count"], x["total"]))
        winning_platform = active_platforms[0]

        return {
            "item_analysis": item_analysis,                      # Individual item price breakdown
            "platform_totals": active_platforms,                   # Total cost per platform
            "recommended_platform": winning_platform["platform"], # Best single store option
            "recommended_basket": winning_platform["items"],       # Basket items for winning store
            "split_cart_total": round(split_cart_total, 2),       # Optimal total if buying item-by-item across stores
            "total_requested": len(items)
        }

    except Exception as e:
        return {
            "item_analysis": [],
            "platform_totals": [],
            "recommended_platform": "",
            "recommended_basket": [],
            "split_cart_total": 0.0,
            "error": str(e)
        }# --- Define the Graph ---
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
