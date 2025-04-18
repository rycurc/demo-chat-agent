"""
TrendChat — agents‑sdk edition (semantic multi‑search, dedup by _id)
pip install openai-agents httpx
"""

from __future__ import annotations
import asyncio
import re
from typing import List, Dict
import httpx
from agents import Agent, Runner, ModelSettings, function_tool

# ---------- Semantic multi‑search tool ----------
API_LIMIT = 20
GLYSTN_ENDPOINT = "https://app.glystn.com/api/transcript_vector_search"

def _clean(text: str) -> str:
    """Remove URLs, compress whitespace, truncate."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000]

@function_tool
async def semantic_multi_search(terms: List[str]) -> List[Dict[str, str]]:
    """
    Accepts ≤3 *different* semantic phrases.
    Returns a list of unique {id, text} dicts (deduped on _id).
    """
    seen_ids: set[str] = set()
    merged: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=15) as client:
        for term in terms[:3]:                            # enforce hard cap
            params = {"term": term, "limit": API_LIMIT}
            try:
                r = await client.get(GLYSTN_ENDPOINT, params=params)
                r.raise_for_status()
                data = r.json().get("results", [])
            except Exception:
                continue

            for item in data:
                _id = item.get("_id")
                text = item.get("transcript")
                if _id and text and _id not in seen_ids:
                    merged.append({"id": _id, "text": _clean(text)})
                    seen_ids.add(_id)
    return merged

# ---------- Agent instructions ----------
INSTRUCTIONS = """
You are glystn assistant, an assistant that uncovers social‑media trends.

• Decide first if a semantic search is needed.  
• If so, call `semantic_multi_search` **once** with up to THREE
  *meaning‑rich* phrases that cover DIFFERENT angles of the user's request.
  – Phrases may be multi‑word and need not be concise; their job is to
    match the wording authors might use in transcripts.
• Receive the merged, de‑duplicated results (each a {id, text} pair).
• Analyse ALL posts and answer the user's question directly in ≤ 150 words.
  – Feel free to cite examples or patterns, but do NOT reveal raw IDs or
    transcripts verbatim.
• Return the answer as plain text; nothing else.
"""

glystn_agent = Agent(
    name="Glystn",
    instructions=INSTRUCTIONS,
    tools=[semantic_multi_search],
    model="o4-mini"
)

# ---------- Summary‑only memory ----------
history: list[dict[str, str]] = []   # stores only assistant summaries

async def chat_loop() -> None:
    print("\n🌀  Glystn Agent – explore social trends (type 'exit' to quit).")
    while True:
        try:
            user = input("\n👤  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Bye!"); break
        if not user or user.lower() in {"exit", "quit", "bye"}:
            print("👋  Bye!"); break

        convo = history + [{"role": "user", "content": user}]
        result = await Runner.run(glystn_agent, convo)
        answer = result.final_output
        print(f"\n🤖  Bot: {answer}")

        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    asyncio.run(chat_loop())
