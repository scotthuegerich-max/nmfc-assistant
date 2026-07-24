"""
main.py

FastAPI skeleton for the NMFC code suggestion API.

Pipeline per request:
  1. Compute density deterministically (density.py) — no AI, no latency cost.
  2. Retrieve top-K candidate commodities via embedding similarity (build_embeddings.py).
  3. Pass user input + density + candidates to Claude for ranking, confidence,
     and rationale (or a single clarifying question if confidence is low).

Run locally:
    export ANTHROPIC_API_KEY=your_key_here
    export NMFC_API_KEY=some_demo_key_for_your_own_auth
    uvicorn main:app --reload

Test:
    curl -X POST http://localhost:8000/v1/nmfc/suggest \
      -H "X-API-Key: some_demo_key_for_your_own_auth" \
      -H "Content-Type: application/json" \
      -d '{
            "description": "wooden dining table, unassembled, in cardboard box",
            "length_in": 48, "width_in": 30, "height_in": 12, "weight_lbs": 65,
            "packaging": "boxed", "palletized": false, "stackable": true, "hazmat": false
          }'
"""

import os
import json
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic

from density import compute_density_pcf, resolve_class
from build_embeddings import search, load_dataset

app = FastAPI(title="NMFC Code Suggestion API", version="0.1.0")

# Demo-only: allows the standalone HTML frontend (opened as a local file, a different
# "origin" than the API) to call this endpoint from the browser. Lock this down to
# your actual frontend's domain before this touches any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_MODEL = "claude-sonnet-4-6"
API_KEY = os.environ.get("NMFC_API_KEY")  # demo-level auth only; swap for real key management in production

SYSTEM_PROMPT = """You are an NMFC freight classification assistant. You will be given:
1. A user's raw description of an item to ship
2. Computed shipment metadata (density, packaging, palletized, stackable, hazmat)
3. A list of candidate commodity records retrieved from the reference dataset, each already
   resolved to a specific class given the computed density (or a fixed class if not density-based)
4. Optionally, a previously asked clarifying question and the user's answer to it — if present,
   treat the answer as authoritative new information about the item, not as a separate item.
   Re-rank and re-score using the original description AND the clarification answer together.
   Only ask another clarifying question if genuine ambiguity remains after incorporating the
   answer — do not ask a second round of questions for minor residual uncertainty.

Your job:
- Rank the candidates by how well they match the user's actual item description and metadata
- Assign a confidence score (0-1) per candidate based on description match quality and any ambiguity
- Write a one-sentence rationale per candidate explaining why it fits (or doesn't fully fit)
- Deprioritize any "NOI" (Not Otherwise Indicated) catch-all commodity below more specific matches
  unless no specific commodity scores reasonably close
- If a candidate is flagged hazmat, do not present it as a confident auto-suggestion — surface it
  but note that hazmat items require manual compliance review regardless of classification confidence
- If the top candidate's confidence is below 0.5, set needs_clarification to true and ask exactly
  ONE targeted question that would most improve confidence — never ask multiple questions

Respond ONLY with valid JSON matching this schema, no markdown, no preamble, no code fences:
{
  "suggestions": [
    { "nmfc_item_id": string, "commodity_description": string, "suggested_class": number, "confidence": number, "rationale": string, "hazmat_flag": boolean }
  ],
  "needs_clarification": boolean,
  "clarifying_question": string | null
}
"""


class SuggestRequest(BaseModel):
    description: str = Field(..., description="Free-text description of the item being shipped")
    length_in: float
    width_in: float
    height_in: float
    weight_lbs: float
    packaging: Optional[str] = None
    palletized: Optional[bool] = None
    stackable: Optional[bool] = None
    hazmat: Optional[bool] = False
    previous_clarifying_question: Optional[str] = None
    clarification_answer: Optional[str] = None


class SuggestResponse(BaseModel):
    computed_density_pcf: float
    suggestions: list
    needs_clarification: bool
    clarifying_question: Optional[str]


def verify_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def build_candidate_payload(request: SuggestRequest, top_k: int = 5) -> tuple[float, list[dict]]:
    """
    Runs the deterministic + retrieval steps and assembles the candidate list
    that gets handed to Claude for reasoning.
    """
    density_result = compute_density_pcf(
        length_in=request.length_in,
        width_in=request.width_in,
        height_in=request.height_in,
        weight_lbs=request.weight_lbs,
    )

    retrieval_query = request.description
    if request.clarification_answer:
        retrieval_query = f"{request.description}. {request.clarification_answer}"

    retrieved = search(retrieval_query, top_k=top_k)
    dataset_by_id = {c["item_id"]: c for c in load_dataset()}

    candidates = []
    for r in retrieved:
        entry = dataset_by_id[r["item_id"]]
        classification = resolve_class(entry, density_result.density_pcf)
        candidates.append(
            {
                "item_id": entry["item_id"],
                "commodity_description": entry["commodity_description"],
                "similarity_score": r["similarity"],
                "resolved_class": classification["class"],
                "resolution_rule": classification["rule"],
                "packaging_notes": entry.get("packaging_notes"),
                "hazmat_flag": entry.get("hazmat_flag", False),
            }
        )

    return density_result.density_pcf, candidates


def call_claude_for_ranking(request: SuggestRequest, density_pcf: float, candidates: list[dict]) -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    user_payload = {
        "user_description": request.description,
        "shipment_metadata": {
            "computed_density_pcf": density_pcf,
            "packaging": request.packaging,
            "palletized": request.palletized,
            "stackable": request.stackable,
            "hazmat": request.hazmat,
        },
        "candidates": candidates,
    }

    if request.clarification_answer:
        user_payload["previous_clarifying_question"] = request.previous_clarifying_question
        user_payload["clarification_answer"] = request.clarification_answer

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
    )

    raw_text = response.content[0].text.strip()
    # Defensive: strip markdown fences if the model adds them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    return json.loads(raw_text)


@app.post("/v1/nmfc/suggest", response_model=SuggestResponse)
def suggest_nmfc_code(request: SuggestRequest, x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)

    try:
        density_pcf, candidates = build_candidate_payload(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        claude_result = call_claude_for_ranking(request, density_pcf, candidates)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Reasoning layer returned malformed JSON")

    return SuggestResponse(
        computed_density_pcf=density_pcf,
        suggestions=claude_result.get("suggestions", []),
        needs_clarification=claude_result.get("needs_clarification", False),
        clarifying_question=claude_result.get("clarifying_question"),
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
