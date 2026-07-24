# NMFC Assistant

An AI-assisted NMFC freight classification API for LTL shipping. Given a shipper's raw item description and shipment details, it returns ranked, confidence-scored NMFC code suggestions with plain-English rationale — reducing the misclassification rebills that happen when shippers guess at a code they don't actually know.

**This repo ships an API, not an app.** The included `demo.html` is a reference client that exercises the API end to end so you can see the classification logic working — it is not the deliverable. The intended integration point is `POST /v1/nmfc/suggest`, meant to be embedded directly into a brokerage's existing shipment-order flow (TMS, quoting tool, booking portal) as a required-field assist, with brokerage's own UI calling this endpoint instead of the demo shown here.

## The problem

LTL freight class determines the rate a shipment is billed at. Most shippers know their item's dimensions and weight; very few know its exact NMFC classification. When the code field is left blank or guessed incorrectly at booking, carriers reclassify and rebill after pickup — a slow, costly, adversarial process for both the shipper and the brokerage managing the account.

Making NMFC code a required field at booking closes that gap, but only if getting it right is nearly as fast as leaving it blank. That's the actual product constraint this API is built around: minimize funnel friction while still producing a defensible, explainable classification.

## Architecture

Three layers, only one of which involves an LLM:

```
shipment input
      │
      ▼
┌─────────────────────┐
│ 1. Density calc      │  deterministic math, no AI, no latency cost
│    (density.py)      │  weight ÷ (L×W×H / 1728) = density (pcf)
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 2. Semantic retrieval│  embed item description, cosine-similarity
│  (build_embeddings.py)│ search against the commodity reference set
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 3. Claude reasoning   │  ranks candidates, scores confidence,
│    (main.py)          │  writes rationale, resolves class per
│                       │  candidate's density table or fixed class
└─────────┬────────────┘
          ▼
  ranked suggestions +
  confidence + rationale
  (+ one clarifying
   question if confidence
   is too low to trust)
```

Density is computed before the LLM ever sees the request, and each candidate's class is already resolved against that density before Claude ranks anything — Claude is reasoning over pre-classified candidates, not doing the arithmetic itself. This keeps the one part of the system with real financial consequences (the class-to-density mapping) fully deterministic and testable, and reserves the LLM for what it's actually good at: matching ambiguous free text to the right commodity, explaining why, and knowing when it doesn't know.

## API

**Request** — `POST /v1/nmfc/suggest`
```json
{
  "description": "wooden dining table, unassembled, in cardboard box",
  "length_in": 48, "width_in": 30, "height_in": 12, "weight_lbs": 65,
  "packaging": "boxed", "palletized": false, "stackable": true, "hazmat": false
}
```

**Response**
```json
{
  "computed_density_pcf": 6.5,
  "suggestions": [
    {
      "nmfc_item_id": "SAMPLE-00101",
      "commodity_description": "Furniture, wooden, unassembled, boxed",
      "suggested_class": 100,
      "confidence": 0.88,
      "rationale": "Directly matches the unassembled wooden furniture description; density of 6.5 pcf resolves to class 100.",
      "hazmat_flag": false
    }
  ],
  "needs_clarification": false,
  "clarifying_question": null
}
```

When top confidence falls below ~0.5, `needs_clarification` flips to `true` with exactly one targeted follow-up question — never a form, never multiple questions. A second call with `clarification_answer` (and `previous_clarifying_question`) set re-ranks using the original description plus the new detail, closing the loop without the caller having to reconstruct anything from scratch.

Full request/response schema is in `main.py`; interactive docs are auto-generated at `/docs` when the API is running (Swagger UI, via FastAPI).

## Key design decisions

- **One clarifying question, never a form.** Every additional required field before a confident suggestion is available is friction a shipper will abandon. The system asks the single most confidence-improving question and stops.
- **NOI (catch-all) is always the fallback, never the leader.** The reasoning layer is explicitly instructed to deprioritize the generic "Not Otherwise Indicated" commodity below any reasonably-scoring specific match — real rebills often trace back to a shipment defaulting into NOI when a cheaper, more specific classification actually applied.
- **Hazmat is surfaced, never auto-confirmed.** A hazmat-flagged candidate is still shown and ranked, but the response explicitly notes it requires manual compliance review regardless of confidence score — this is a liability question, not a convenience one.
- **Deterministic math stays deterministic.** Density-to-class resolution never goes through the LLM. This makes the one legally/financially sensitive calculation in the pipeline auditable and unit-testable independent of model behavior.
- **The clarification loop is stateless by design.** No session or database is required to close the ask → answer → re-rank cycle — the caller resends the original payload plus the answer. Simpler to integrate into an existing booking flow that may not want to manage conversation state for a single field.

## Data source — read before treating this as production-ready

`sample_nmfc_dataset.json` is a synthetic, illustrative commodity set built for this demo. **It is not the real NMFC tariff.** The actual National Motor Freight Classification is copyrighted and licensed through the National Motor Freight Traffic Association (NMFTA); a production deployment of this system would require licensing the full commodity dataset from NMFTA and replacing this file with it. The architecture — density calculator, embedding retrieval, reasoning layer, API contract — is unaffected by that swap; only the data file changes.

## Running it locally

```bash
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install sentence-transformers fastapi anthropic uvicorn
export ANTHROPIC_API_KEY=your_key                  # Windows: set ANTHROPIC_API_KEY=your_key
export NMFC_API_KEY=your_own_demo_key

python build_embeddings.py     # builds the retrieval cache once
uvicorn main:app --reload      # starts the API at localhost:8000
```

Open `demo.html` directly in a browser to exercise the API through the reference client, or hit `/docs` for the interactive Swagger UI, or call `/v1/nmfc/suggest` directly from your own client.

## What a real brokerage integration would need beyond this repo

- NMFTA-licensed commodity data in place of the sample set
- Multi-tenant API key management and rate limiting (current auth is a single shared demo key)
- Persistent logging of overrides — when a user picks a different code than the top suggestion, that's a feedback signal worth capturing to improve retrieval quality over time
- A move from the local sentence-transformer model to a hosted embedding provider (e.g. Voyage AI) if this needs to scale beyond a single-instance deployment
- Webhook or SDK wrappers so a brokerage's TMS can call this as a drop-in step in their existing booking flow, rather than the standalone reference UI included here

## Stack

Python, FastAPI, Claude (Anthropic API) for the reasoning layer, `sentence-transformers` for local embeddings, vanilla HTML/CSS/JS for the reference client.
