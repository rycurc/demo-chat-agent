from __future__ import annotations
import asyncio
import re
from typing import List, Dict

import httpx
import streamlit as st
from agents import Agent, Runner, function_tool

# ---------------- Semantic multi‑search tool ----------------
API_LIMIT = 25
GLYSTN_ENDPOINT = "https://app.glystn.com/api/transcript_vector_search"

def _clean(text: str) -> str:
    """Remove URLs, compress whitespace, truncate."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000]

@function_tool
async def semantic_search(term: str) -> List[Dict[str, str]]:
    """
    Accepts a single semantic phrase.
    Returns a list of unique {id, text} dicts.
    """
    results: list[dict[str, str]] = []
    print("\n 🔍 Searching for:", term)

    async with httpx.AsyncClient(timeout=15) as client:
        params = {"term": term, "limit": API_LIMIT}
        try:
            r = await client.get(GLYSTN_ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json().get("results", [])
        except Exception as e:
            print(f"Error during semantic search: {e}") # Add some basic error logging
            return [] # Return empty list on error

        for item in data:
            _id = item.get("_id")
            text = item.get("transcript")
            if _id and text: # No need to check seen_ids
                results.append({"id": _id, "text": _clean(text)})

    return results

# ---------------- Agent instructions ----------------
INSTRUCTIONS = """
You are glystn assistant, an assistant that uncovers social‑media trends. Respond as a research assistant.

• Decide first if a semantic search is needed to find social media posts that match the user's request.  
• If so, call `semantic_search` **once** with a *meaning‑rich* phrase  
  – Phrases may be multi‑word and need not be concise; their job is to
    match the wording authors might use in transcripts.
• Receive the results (each a {id, text} pair).
• Analyse ALL posts and answer the user's question directly in ≤ 150 words.  
  – Feel free to cite examples or patterns, but do NOT reveal raw IDs or
    transcripts verbatim. 
  - It's good to show the user snippets from the transcripts that are relevant to the question and match the trends. 
• Return the answer as plain text; nothing else.
- ONLY answer with the information from the transcripts. 
- Refer to the people who created the content as 'creators' -- the trends are coming from creators.  
"""

glystn_agent = Agent(
    name="Glystn",
    instructions=INSTRUCTIONS,
    tools=[semantic_search],
    model="o4-mini",
)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Glystn Assistant", page_icon="💬")
st.title("Glystn Assistant")

# Summary‑only memory held in session_state
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input("Ask me about social‑media trends…")
if prompt:
    with st.spinner("Thinking…"):
        convo = st.session_state.history + [
            {"role": "user", "content": prompt}
        ]
        # Runner.run returns an object whose final_output is the agent's reply
        answer = asyncio.run(Runner.run(glystn_agent, convo)).final_output

        # Persist the conversation
        st.session_state.history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])

# Render the full conversation each rerun
for turn in st.session_state.history:
    st.chat_message(turn["role"]).write(turn["content"])
