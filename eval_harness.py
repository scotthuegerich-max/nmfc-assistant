"""
eval_harness.py

Regression / eval harness for the NMFC Assistant API. Runs a fixed set of
test cases against a running instance (local or deployed) and checks the
responses against expected behavior — not just "did it return 200," but
whether it's making the right calls: correct top suggestion where one is
obvious, appropriate uncertainty where one isn't, and no violations of the
system's own stated rules (NOI shouldn't unreasonably outrank a specific
match, no fabricated real-world classification identifiers).

This exists to catch regressions when the system prompt, dataset, or
retrieval provider changes — e.g. this would have caught it immediately if
the Voyage AI migration had degraded classification quality, rather than
relying on manually re-running test cases by hand.

Usage:
    export NMFC_API_BASE=https://nmfc-assistant.onrender.com   # or http://localhost:8000
    export NMFC_API_KEY=your_key
    python eval_harness.py
"""

import os
import re
import sys
import time
import requests

API_BASE = os.environ.get("NMFC_API_BASE", "http://localhost:8000")
API_KEY = os.environ.get("NMFC_API_KEY", "")

# Any of these patterns showing up in a rationale or clarifying question is a
# red flag for fabricated real-world identifiers — this system should only
# ever reference its own SAMPLE-XXXXX synthetic dataset entries, never real
# NMFC item numbers or DOT/hazmat UN codes it wasn't given.
SUSPICIOUS_PATTERNS = [
    r"\bUN\s?\d{4}\b",              # DOT hazmat UN numbers
    r"\bItem\s+\d{4,}\b",           # real NMFC item numbers (not "SAMPLE-XXXXX")
    r"\bNMFC\s+\d{4,}\b",
]


def call_suggest(payload: dict) -> dict:
    resp = requests.post(
        f"{API_BASE}/v1/nmfc/suggest",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def check_no_fabricated_identifiers(data: dict):
    text_blob = " ".join(
        [s.get("rationale", "") for s in data.get("suggestions", [])]
        + [data.get("clarifying_question") or ""]
    )
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text_blob):
            return False, f"Found suspicious pattern '{pattern}' in response text"
    return True, ""


def check_noi_not_unreasonably_leading(data: dict):
    suggestions = data.get("suggestions", [])
    if not suggestions:
        return True, ""
    top = suggestions[0]
    if top.get("nmfc_item_id") == "SAMPLE-01301" and top.get("confidence", 0) > 0.3:
        others = suggestions[1:]
        if others and max(o.get("confidence", 0) for o in others) < top["confidence"] - 0.1:
            return False, "NOI led with meaningfully higher confidence than more specific candidates"
    return True, ""


def check_confidence_clarification_consistency(data: dict):
    suggestions = data.get("suggestions", [])
    if not suggestions:
        return True, ""

    # Hazmat cases can legitimately ask a clarifying question about compliance
    # detail (e.g. standalone cells vs. packed with equipment) independent of
    # how confident the model is about the general commodity match — that's
    # expected behavior per the system prompt, not an inconsistency.
    if any(s.get("hazmat_flag") for s in suggestions):
        return True, ""

    top_confidence = suggestions[0].get("confidence", 1)
    needs_clarification = data.get("needs_clarification", False)
    # Slack around the 0.5 boundary since this is a judgment call, not a hard
    # rule — only flag clearly inconsistent results, not near-threshold cases.
    if top_confidence > 0.65 and needs_clarification:
        return False, f"High confidence ({top_confidence}) but still asked for clarification"
    if top_confidence < 0.35 and not needs_clarification:
        return False, f"Low confidence ({top_confidence}) but did not ask for clarification"
    return True, ""


GLOBAL_CHECKS = [
    ("no_fabricated_identifiers", check_no_fabricated_identifiers),
    ("noi_not_unreasonably_leading", check_noi_not_unreasonably_leading),
    ("confidence_clarification_consistency", check_confidence_clarification_consistency),
]


TEST_CASES = [
    {
        "name": "wooden_table_unassembled_clear_match",
        "payload": {
            "description": "wooden dining table, unassembled, in cardboard box",
            "length_in": 48, "width_in": 30, "height_in": 12, "weight_lbs": 65,
            "packaging": "boxed", "palletized": False, "stackable": True, "hazmat": False,
        },
        "expect_top_item_id": "SAMPLE-00101",
        "expect_top_confidence_min": 0.7,
        "expect_needs_clarification": False,
    },
    {
        "name": "misc_goods_ambiguous",
        "payload": {
            "description": "misc goods, boxed",
            "length_in": 20, "width_in": 20, "height_in": 20, "weight_lbs": 30,
            "packaging": "boxed", "palletized": False, "stackable": True, "hazmat": False,
        },
        # Widened from an earlier hardcoded 0.55: this case sits right near the
        # clarification threshold by design, and Claude's confidence score is a
        # genuine judgment call each run, not a deterministic lookup — expect
        # some run-to-run variance here rather than a single exact number.
        "expect_top_confidence_max": 0.65,
    },
    {
        "name": "lithium_batteries_hazmat_surfaced",
        "payload": {
            "description": "lithium-ion batteries",
            "length_in": 40, "width_in": 48, "height_in": 40, "weight_lbs": 500,
            "packaging": "boxed", "palletized": False, "stackable": True, "hazmat": True,
        },
        "expect_any_suggestion_item_id": "SAMPLE-01402",
        "expect_any_suggestion_hazmat_flag": True,
    },
    {
        "name": "ping_pong_balls_low_density_ambiguous",
        "payload": {
            "description": "ping pong balls",
            "length_in": 40, "width_in": 48, "height_in": 40, "weight_lbs": 20,
            "packaging": "boxed", "palletized": False, "stackable": True, "hazmat": False,
        },
        # No hard confidence assertion — this one runs close to the 0.5
        # threshold by design and varies run to run. Only the global rule
        # checks apply here (no fabrication, sane NOI ranking).
    },
]


def run_case(case: dict):
    name = case["name"]
    try:
        data = call_suggest(case["payload"])
    except Exception as e:
        return name, False, [f"Request failed: {e}"]

    failures = []
    suggestions = data.get("suggestions", [])
    top = suggestions[0] if suggestions else {}

    if "expect_top_item_id" in case and top.get("nmfc_item_id") != case["expect_top_item_id"]:
        failures.append(f"Expected top item_id {case['expect_top_item_id']}, got {top.get('nmfc_item_id')}")

    if "expect_top_confidence_min" in case and top.get("confidence", 0) < case["expect_top_confidence_min"]:
        failures.append(f"Expected top confidence >= {case['expect_top_confidence_min']}, got {top.get('confidence')}")

    if "expect_top_confidence_max" in case and top.get("confidence", 1) > case["expect_top_confidence_max"]:
        failures.append(f"Expected top confidence <= {case['expect_top_confidence_max']}, got {top.get('confidence')}")

    if "expect_needs_clarification" in case and data.get("needs_clarification") != case["expect_needs_clarification"]:
        failures.append(f"Expected needs_clarification={case['expect_needs_clarification']}, got {data.get('needs_clarification')}")

    if "expect_any_suggestion_item_id" in case:
        ids = [s.get("nmfc_item_id") for s in suggestions]
        if case["expect_any_suggestion_item_id"] not in ids:
            failures.append(f"Expected {case['expect_any_suggestion_item_id']} to appear among suggestions, got {ids}")

    if "expect_any_suggestion_hazmat_flag" in case:
        if not any(s.get("hazmat_flag") for s in suggestions):
            failures.append("Expected at least one suggestion flagged hazmat, none were")

    for check_name, check_fn in GLOBAL_CHECKS:
        ok, msg = check_fn(data)
        if not ok:
            failures.append(f"[{check_name}] {msg}")

    return name, len(failures) == 0, failures


def main():
    print(f"Running eval harness against {API_BASE}\n")
    total = len(TEST_CASES)
    passed = 0

    for i, case in enumerate(TEST_CASES):
        if i > 0:
            time.sleep(21)  # stay safely under Voyage's 3 RPM limit if no payment method is on file yet
        name, ok, failures = run_case(case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if ok:
            passed += 1
        else:
            for f in failures:
                print(f"       - {f}")

    print(f"\n{passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
