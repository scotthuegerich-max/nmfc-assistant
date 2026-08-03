"""
sheets_logger.py

Appends one row per user selection to a Google Sheet, so overrides (when a
user picks something other than the top suggestion) can be reviewed later
as a feedback signal for improving retrieval/reasoning quality.

Setup (one-time):
  1. Create a Google Cloud service account, enable the Sheets + Drive APIs,
     download its JSON key file.
  2. Share your target Google Sheet with the service account's email
     (Editor access) — same as sharing with any person.
  3. Set these environment variables:
       GOOGLE_SHEETS_CREDENTIALS_PATH = path to the downloaded JSON key file
       GOOGLE_SHEET_ID                = the ID from the sheet's URL
       GOOGLE_SHEET_WORKSHEET         = optional, defaults to "Overrides"

This module fails loudly on misconfiguration (missing env vars, bad creds)
but is called in a try/except from main.py — a logging failure should never
break the classification response itself.
"""

import os
import json
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

CREDENTIALS_PATH = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_PATH")
CREDENTIALS_JSON = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON")  # for cloud deploys — see README
SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
WORKSHEET_NAME = os.environ.get("GOOGLE_SHEET_WORKSHEET", "Overrides")

HEADER = [
    "timestamp_utc",
    "item_description",
    "computed_density_pcf",
    "top_suggested_item_id",
    "top_suggested_class",
    "top_confidence",
    "user_selected_item_id",
    "user_selected_class",
    "user_selected_confidence",
    "was_override",
]


def _get_worksheet():
    if not SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID must be set as an environment variable before logging can work.")
    if not CREDENTIALS_PATH and not CREDENTIALS_JSON:
        raise RuntimeError(
            "Either GOOGLE_SHEETS_CREDENTIALS_PATH (local dev, path to a JSON key file) or "
            "GOOGLE_SHEETS_CREDENTIALS_JSON (cloud deploy, the key file's contents as a string) "
            "must be set as an environment variable before logging can work."
        )

    if CREDENTIALS_JSON:
        creds = Credentials.from_service_account_info(json.loads(CREDENTIALS_JSON), scopes=SCOPES)
    else:
        creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)

    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)

    try:
        worksheet = sheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=len(HEADER))

    # Add the header row if the sheet/worksheet is brand new and empty
    if not worksheet.row_values(1):
        worksheet.append_row(HEADER)

    return worksheet


def log_selection(row: dict) -> None:
    """
    row should contain keys matching (a subset of) HEADER; missing keys are
    written as blank cells. Column order is fixed regardless of dict order.
    """
    worksheet = _get_worksheet()
    ordered_row = [row.get(col, "") for col in HEADER]
    worksheet.append_row(ordered_row, value_input_option="USER_ENTERED")


def build_log_row(
    description: str,
    computed_density_pcf: float,
    top_suggestion: dict | None,
    selected_item_id: str,
    selected_suggestion: dict | None,
) -> dict:
    top_id = top_suggestion["nmfc_item_id"] if top_suggestion else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "item_description": description,
        "computed_density_pcf": computed_density_pcf,
        "top_suggested_item_id": top_id,
        "top_suggested_class": top_suggestion["suggested_class"] if top_suggestion else "",
        "top_confidence": top_suggestion["confidence"] if top_suggestion else "",
        "user_selected_item_id": selected_item_id,
        "user_selected_class": selected_suggestion["suggested_class"] if selected_suggestion else "",
        "user_selected_confidence": selected_suggestion["confidence"] if selected_suggestion else "",
        "was_override": selected_item_id != top_id,
    }
