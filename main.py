import json
import os
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")
if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY in environment.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

app = FastAPI(title="Scout Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


SYSTEM_PROMPT = """
You are Scout, a research agent.
You must ONLY use the provided web search results and never rely on your internal memory.

You should reason about whether the current evidence is enough.
If you need more information, return a JSON object:
{
  "needs_more_info": true,
  "refined_query": "a better follow-up search query",
  "thought": "brief reason for what is missing"
}

If you have enough information, return a JSON object:
{
  "needs_more_info": false,
  "thought": "brief reason for why evidence is sufficient",
  "final_answer": "Structured research brief with headings: Summary, Key Findings (3-5 bullets), and Implications."
}
""".strip()


def _extract_search_snippets(search_response: Any) -> List[str]:
    if isinstance(search_response, dict):
        results = search_response.get("results", [])
    elif isinstance(search_response, list):
        results = search_response
    else:
        results = []

    snippets: List[str] = []
    for item in results:
        if isinstance(item, dict):
            content = item.get("content")
            if content:
                snippets.append(str(content))
    return snippets


def _run_search(query: str) -> List[str]:
    try:
        response = tavily_client.search(query)
        return _extract_search_snippets(response)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Tavily search failed: {exc}") from exc


def _llm_reason(query: str, snippets: List[str]) -> Dict[str, Any]:
    context = "\n\n".join(f"- {snippet}" for snippet in snippets[:10]) or "No search results found."
    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Search evidence:\n{context}\n\n"
        "Respond with valid JSON only."
    )

    payload = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq completion failed: {exc}") from exc

    raw = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Groq: {raw}") from exc


@app.post("/research")
def research(request: ResearchRequest) -> Dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    steps: List[Dict[str, Any]] = []
    current_query = query
    final_answer = ""
    max_iterations = 3

    for iteration in range(1, max_iterations + 1):
        thought = f"Iteration {iteration}: gather evidence for query '{current_query}'."
        search_snippets = _run_search(current_query)

        llm_output = _llm_reason(query=query, snippets=search_snippets)
        needs_more_info = bool(llm_output.get("needs_more_info"))
        llm_thought = str(llm_output.get("thought", "")).strip() or "No thought provided."

        action = {"type": "search", "query": current_query}
        observation = {
            "result_count": len(search_snippets),
            "sample_results": search_snippets[:3],
            "llm_assessment": llm_thought,
            "needs_more_info": needs_more_info,
        }

        steps.append(
            {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        )

        if not needs_more_info:
            final_answer = str(llm_output.get("final_answer", "")).strip()
            if not final_answer:
                final_answer = (
                    "Summary\nInsufficient final answer formatting from model.\n\n"
                    "Key Findings\n- Evidence was gathered but no final brief was returned.\n\n"
                    "Implications\nAdditional prompt tuning may be required."
                )
            break

        refined_query = str(llm_output.get("refined_query", "")).strip()
        if not refined_query:
            final_answer = (
                "Summary\nResearch ended before sufficient evidence was synthesized.\n\n"
                "Key Findings\n- The model requested more information but did not provide a refined query.\n\n"
                "Implications\nA better follow-up query strategy is needed."
            )
            break
        current_query = refined_query

    if not final_answer:
        final_pass = _llm_reason(query=query, snippets=_run_search(current_query))
        final_answer = str(final_pass.get("final_answer", "")).strip()
        if not final_answer:
            final_answer = (
                "Summary\nMaximum search iterations reached.\n\n"
                "Key Findings\n- Additional evidence may still be needed.\n\n"
                "Implications\nConsider raising iteration limit or improving query refinement."
            )

    return {"steps": steps, "final_answer": final_answer}