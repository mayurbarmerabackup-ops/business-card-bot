"""
Business Card Bot  (v2)
=======================
Telegram photo of a business card
    -> Gemini (vision) reads it
    -> duplicate check (same person already in the sheet?)
    -> new row in Google Sheets
    -> a .vcf contact file sent back in the chat (tap to add to iPhone / Android)

Runs on Vercel as a standard Python web app: this file must be called app.py, sit at the
top level of the repository, and expose a variable named `app` (that is what Vercel looks
for). Telegram calls the site (a "webhook") every time you message the bot.

Environment variables (set in the Vercel dashboard):

  TELEGRAM_BOT_TOKEN           token from @BotFather
  TELEGRAM_WEBHOOK_SECRET      any random string you invent (A-Z a-z 0-9 _ -)
  ALLOWED_USER_IDS             your numeric Telegram user id(s), comma separated
  GEMINI_API_KEY               key from aistudio.google.com/api-keys
  GOOGLE_SERVICE_ACCOUNT_JSON  full contents of the service-account .json key
  GOOGLE_SHEET_ID              the long id inside your Google Sheet URL
  WORKSHEET_NAME               (optional) tab name, default "Cards"
  DEFAULT_PHONE_REGION         (optional) 2-letter country code used for phone
                               numbers printed WITHOUT a country code when the
                               card's own country can't be worked out. Default "IN".
  GEMINI_MODEL                 (optional) default "gemini-3.8-flash"
  GEMINI_FALLBACK_MODEL        (optional) used if the main model is busy/slow. Default "gemini-3.5-flash-lite"
  GEMINI_THINKING              (optional) minimal | low | medium | high. Default "low" (fastest sensible)
"""

import hmac
import io
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from http import HTTPStatus
from typing import List, Literal

import gspread
import phonenumbers
import requests
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash-lite"

# Telegram stops waiting for our reply after about a minute and then SENDS THE SAME MESSAGE AGAIN,
# which used to cause double processing. So everything must finish well inside that time.
REQUEST_BUDGET_SECONDS = 50
RESERVE_SECONDS = 8              # kept free for re-checking/writing the sheet, sending the vCard, replying
FIRST_ATTEMPT_MAX_SECONDS = 25   # max wait on the main model before switching to the backup model

# Row 1 of the sheet must be exactly these headers, in this order (columns A-P).
# You may add your own extra columns to the RIGHT of column P.
COLUMNS = [
    "Timestamp (UTC)",
    "Business Name",
    "Business Name (English)",
    "Contact Person",
    "Contact Person (English)",
    "Job Title",
    "Phone Numbers",
    "Email",
    "Website",
    "Address",
    "City",
    "Country",
    "Card Language",
    "Miscellaneous",
    "Added By",
    "Telegram File ID",  # used to spot the same photo sent twice
]
COL = {name: i for i, name in enumerate(COLUMNS)}

HELP_TEXT = (
    "👋 Send me a photo of a business card and I'll:\n"
    "1) save its details to your Google Sheet\n"
    "2) send you a contact file (.vcf) you can tap to add to your phone\n\n"
    "• Works with cards in any language\n"
    "• One card per photo\n"
    "• If I already have that person, I'll tell you instead of adding them twice. "
    "To save anyway, send the photo again with the caption: force\n"
    "• For small print, send the photo as a File (paperclip → File) instead of a "
    "normal photo — Telegram shrinks normal photos."
)


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if isinstance(value, str):
        value = value.strip()  # ignore accidental spaces / line breaks around a pasted value
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()  # ...and quote marks wrapped around it
    if required and not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


# --------------------------------------------------------------------------
# Telegram helpers
# --------------------------------------------------------------------------

def tg_api(method, **payload):
    token = env("TELEGRAM_BOT_TOKEN", required=True)
    r = requests.post(f"https://api.telegram.org/bot{token}/{method}", json=payload, timeout=20)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description')}")
    return data["result"]


def send_message(chat_id, text, reply_to=None):
    payload = {"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True}
    if reply_to:
        payload["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
    try:
        tg_api("sendMessage", **payload)
    except Exception as e:  # never let a failed reply crash the function
        log("sendMessage failed:", e)


def send_document(chat_id, file_bytes, filename, caption, mime="text/vcard"):
    token = env("TELEGRAM_BOT_TOKEN", required=True)
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data={"chat_id": chat_id, "caption": caption[:1000]},
        files={"document": (filename, file_bytes, mime)},
        timeout=40,
    )
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram sendDocument failed: {data.get('description')}")


def download_telegram_file(file_id):
    token = env("TELEGRAM_BOT_TOKEN", required=True)
    info = tg_api("getFile", file_id=file_id)
    r = requests.get(f"https://api.telegram.org/file/bot{token}/{info['file_path']}", timeout=40)
    r.raise_for_status()
    return r.content


def pick_image(msg):
    """Return (file_id, file_unique_id) of the image in a Telegram message, or (None, None)."""
    photos = msg.get("photo")
    if photos:
        best = max(photos, key=lambda p: p.get("file_size") or (p["width"] * p["height"]))
        return best["file_id"], best["file_unique_id"]
    doc = msg.get("document")
    if doc and (doc.get("mime_type") or "").startswith("image/"):
        return doc["file_id"], doc["file_unique_id"]
    return None, None


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------

def prepare_image(raw_bytes):
    """Fix phone-camera rotation, shrink to a sensible size, return JPEG bytes."""
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Gemini vision extraction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a meticulous data-entry assistant who reads photographed business cards.
Return the details of the card as JSON that follows the response schema, using what is printed on the card.

Rules:
- The card can be in ANY language or script (Arabic, Hindi, Chinese, Japanese, Korean, Thai, Cyrillic, etc.), may be bilingual, and the photo may show both sides. Copy text exactly as printed, in its original script.
- For the *_english fields give a Latin-letters / English version (transliterate names, translate or romanise business names). Leave them empty if the card is already in English.
- In phone numbers, convert any non-Western digits (Arabic-Indic, Devanagari, etc.) to ordinary 0-9 digits.
- List EVERY phone number on the card, exactly as printed. Keep a printed "+" and country code. NEVER add a country code that is not printed. Keep extensions with their number. Label each one: mobile, office, fax, whatsapp, toll_free or unknown. Do not repeat the same number twice.
- country_iso2 = the 2-letter ISO country code of the country where this business is located, judged from the address, city, phone prefix, website domain or script. Use an empty string if you cannot tell.
- NEVER guess or invent anything. If something is partly unreadable or you are not sure which field it belongs to, leave it out of the specific field and put it in `miscellaneous`, prefixed with "UNCLEAR:".
- `miscellaneous` = everything useful that does not fit another field: taglines, services/products, social-media handles, registration / GST / licence numbers, extra people, secondary addresses or offices, QR-code text, and anything unclear. Separate items with " | ". Use an empty string if there is nothing.
- If several people are on the card, put the first/main one in contact_person and the others in miscellaneous.
- If the image is not a business card or is unreadable, set is_business_card to false and briefly say what it is in miscellaneous."""


class Phone(BaseModel):
    number: str = Field(description="As printed, 0-9 digits only")
    label: Literal["mobile", "office", "fax", "whatsapp", "toll_free", "unknown"]


class BusinessCard(BaseModel):
    """The structure Gemini must fill in (sent to Gemini as the response schema)."""
    is_business_card: bool
    business_name: str = Field(description="Company / business name as printed")
    business_name_english: str
    contact_person: str = Field(description="Person's name as printed")
    contact_person_english: str
    job_title: str
    phones: List[Phone]
    emails: List[str]
    websites: List[str]
    address: str = Field(description="Full postal address on one line")
    city: str
    country: str = Field(description="Country name in English")
    country_iso2: str = Field(description="2-letter ISO country code, or empty if unknown")
    card_language: str = Field(description="Main language of the card, in English")
    miscellaneous: str


STRING_FIELDS = [
    "business_name", "business_name_english", "contact_person", "contact_person_english",
    "job_title", "address", "city", "country", "country_iso2", "card_language", "miscellaneous",
]


def normalize_card(raw):
    """Make the model's answer safe to use: strings are strings, lists are lists, never None."""
    card = {"is_business_card": raw.get("is_business_card", True) is not False}
    for key in STRING_FIELDS:
        card[key] = str(raw.get(key) or "").strip()
    for key in ("emails", "websites"):
        card[key] = [str(x).strip() for x in (raw.get(key) or []) if str(x or "").strip()]
    phones = []
    for p in raw.get("phones") or []:
        if isinstance(p, dict) and str(p.get("number") or "").strip():
            phones.append({
                "number": str(p["number"]).strip(),
                "label": str(p.get("label") or "unknown").strip() or "unknown",
            })
    card["phones"] = phones
    return card


def is_busy_error(e):
    """True for 'Gemini is overloaded / rate-limited / too slow' style errors (worth trying the backup model)."""
    code = getattr(e, "code", None)
    text = f"{type(e).__name__} {e}".lower()
    return (code in (408, 429, 500, 502, 503, 504) or "timeout" in text or "timed out" in text
            or "overloaded" in text or "high demand" in text or "unavailable" in text)


def gemini_config(thinking_level):
    extra = {}
    level = getattr(types.ThinkingLevel, thinking_level.strip().upper(), None) if thinking_level else None
    if level is not None:
        # Less "thinking" = much faster. Reading a card doesn't need deep reasoning.
        extra["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=BusinessCard,
        max_output_tokens=8192,  # generous: the model's "thinking" tokens count towards this
        **extra,
    )


def parse_gemini_response(response):
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        raw = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    else:
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no card data (the image may have been blocked or unreadable).")
        try:
            raw = json.loads(re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip())
        except json.JSONDecodeError:
            raise RuntimeError("Gemini's answer was cut off or malformed. Please send the photo again.")
    return normalize_card(raw)


def extract_card(image_bytes, deadline=None):
    """Read the card with Gemini. Never waits past `deadline` (a time.monotonic() value).
    Main model first; if it is busy or slow, the backup model gets the remaining time."""
    api_key = env("GEMINI_API_KEY", required=True)
    if deadline is None:
        deadline = time.monotonic() + REQUEST_BUDGET_SECONDS
    primary = env("GEMINI_MODEL", DEFAULT_MODEL)
    backup = env("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
    models = [primary] + ([backup] if backup and backup != primary else [])
    level = env("GEMINI_THINKING", "low")
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Read this business card.",
    ]

    last_error = None
    for position, model in enumerate(models):
        available = deadline - time.monotonic() - RESERVE_SECONDS
        if available < 6:
            break  # not enough time left to try anything else
        is_last = position == len(models) - 1
        timeout = available if is_last else min(available, FIRST_ATTEMPT_MAX_SECONDS)
        use_level = level
        for _ in range(2):  # a 2nd pass happens only if this model refuses the thinking setting
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),  # milliseconds
                )
                response = client.models.generate_content(
                    model=model, contents=contents, config=gemini_config(use_level)
                )
                return parse_gemini_response(response)
            except Exception as e:
                if use_level and getattr(e, "code", None) == 400 and "thinking" in str(e).lower():
                    use_level = ""  # this model doesn't accept that thinking level: retry without it
                    continue
                last_error = e
                if is_busy_error(e):
                    log(f"Gemini model {model} busy/slow: {e}")
                    break  # try the backup model
                raise
    raise RuntimeError(
        "Gemini is busy right now (it did not answer in time). Please send the card again in a minute."
    ) from last_error


# --------------------------------------------------------------------------
# Phone-number clean-up (international, with/without country code)
# --------------------------------------------------------------------------

def to_ascii_digits(text):
    """Turn Arabic-Indic / Devanagari / full-width digits into plain 0-9."""
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in text)


def normalize_phones(phones, card_region, default_region):
    """
    Returns (entries, problems).

    Each entry: {"pretty": "+971 50 123 4567",   # for the sheet / chat
                 "dial":   "+971501234567",      # for the vCard (best for dialling & WhatsApp)
                 "label":  "mobile"}

    - Number printed WITH a country code (+971 ...)  -> used as is.
    - Number printed WITHOUT one                     -> assumed to belong to the card's own
      country (from the address); if that is unknown, DEFAULT_PHONE_REGION is used.
    """
    regions = []
    for r in (card_region, default_region):
        r = (r or "").strip().upper()
        if r and r not in regions:
            regions.append(r)

    entries, problems, seen = [], [], set()
    for item in phones or []:
        raw = to_ascii_digits(item.get("number") or "").strip()
        if not raw:
            continue
        label = (item.get("label") or "").strip()

        valid_number, possible_number = None, None
        for region in regions or [None]:
            try:
                parsed = phonenumbers.parse(raw, region)
            except phonenumbers.NumberParseException:
                continue
            if phonenumbers.is_valid_number(parsed):
                valid_number = parsed
                break
            if possible_number is None and phonenumbers.is_possible_number(parsed):
                possible_number = parsed

        chosen = valid_number or possible_number
        if chosen is not None:
            pretty = phonenumbers.format_number(chosen, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            dial = phonenumbers.format_number(chosen, phonenumbers.PhoneNumberFormat.E164)
            if chosen.extension:
                dial += "," + chosen.extension  # a comma = "pause", then the extension
            key = re.sub(r"\D", "", dial)
            if valid_number is None:
                problems.append(f"UNCLEAR: phone number may be wrong -> {pretty}")
        else:
            pretty = raw
            dial = re.sub(r"[^\d+,]", "", raw) or raw
            key = re.sub(r"\D", "", raw) or raw
            problems.append(f"UNCLEAR: could not standardise phone number -> {raw}")

        if key in seen:
            continue
        seen.add(key)
        entries.append({"pretty": pretty, "dial": dial, "label": label})

    return entries, problems


def phones_cell(entries):
    parts = []
    for e in entries:
        label = e["label"]
        parts.append(f"{e['pretty']} ({label})" if label and label != "unknown" else e["pretty"])
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Google Sheets
# --------------------------------------------------------------------------

def open_sheet():
    """Open the worksheet, make sure row 1 is right, and return (worksheet, all_rows)."""
    # raw_decode reads the first complete JSON object and ignores anything accidentally pasted after it
    info, _ = json.JSONDecoder().raw_decode(env("GOOGLE_SERVICE_ACCOUNT_JSON", required=True))
    if "private_key" in info:  # fixes keys whose line breaks got double-escaped when pasted
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    creds = Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    spreadsheet = gspread.authorize(creds).open_by_key(env("GOOGLE_SHEET_ID", required=True))

    tab = env("WORKSHEET_NAME", "Cards")
    try:
        ws = spreadsheet.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=tab, rows=1000, cols=len(COLUMNS))

    rows = ws.get_all_values()
    if not rows or not any(c.strip() for c in rows[0]):
        ws.append_row(COLUMNS, value_input_option="RAW")
        try:
            ws.format(f"A1:{chr(64 + len(COLUMNS))}1", {"textFormat": {"bold": True}})
            ws.freeze(rows=1)
        except Exception as e:  # cosmetic only
            log("header formatting skipped:", e)
        rows = [list(COLUMNS)]
    else:
        header = [c.strip().casefold() for c in rows[0][: len(COLUMNS)]]
        if header != [c.casefold() for c in COLUMNS]:
            raise RuntimeError(
                "The header row (row 1) of your sheet tab doesn't match what the bot expects. "
                "Row 1, columns A to P, must be exactly: " + ", ".join(COLUMNS)
            )
    return ws, rows


def cell(row, column_name):
    i = COL[column_name]
    return row[i].strip() if i < len(row) else ""


def build_row(card, phones_text, misc_text, added_by, file_unique_id):
    values = {
        "Timestamp (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "Business Name": card.get("business_name", ""),
        "Business Name (English)": card.get("business_name_english", ""),
        "Contact Person": card.get("contact_person", ""),
        "Contact Person (English)": card.get("contact_person_english", ""),
        "Job Title": card.get("job_title", ""),
        "Phone Numbers": phones_text,
        "Email": "; ".join(e.strip() for e in card.get("emails") or [] if e.strip()),
        "Website": "; ".join(w.strip() for w in card.get("websites") or [] if w.strip()),
        "Address": card.get("address", ""),
        "City": card.get("city", ""),
        "Country": card.get("country", ""),
        "Card Language": card.get("card_language", ""),
        "Miscellaneous": misc_text,
        "Added By": added_by,
        "Telegram File ID": file_unique_id,
    }
    return [values[c] for c in COLUMNS]


def append_and_locate(ws, row, fallback_row_number):
    """Append the row; return the sheet row number it landed on."""
    result = ws.append_row(row, value_input_option="RAW", table_range="A1")
    try:
        match = re.search(r"!A(\d+)", result["updates"]["updatedRange"])
        return int(match.group(1))
    except Exception:
        return fallback_row_number


# --------------------------------------------------------------------------
# Duplicate check — "is this person already in the sheet?"
# --------------------------------------------------------------------------
# A card counts as a duplicate when, for some existing row:
#   * it shares an EMAIL address, or
#   * it has the same person NAME *and* the same COMPANY.
# Smart bits:
#   * Names/companies are compared in both the original script and the English
#     version, ignoring case, punctuation, "Mr/Dr", word order, and "Pvt Ltd / LLC / Inc".
#   * A shared mailbox such as info@company.com is NOT treated as a match when both
#     cards clearly name two different people.

HONORIFICS = {"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "eng", "sir", "madam", "shri", "smt", "sh"}
COMPANY_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "corp", "corporation", "co", "company", "pvt", "private",
    "pte", "plc", "gmbh", "llp", "fze", "fzco", "sdn", "bhd", "pty",
}


def _norm(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def norm_name(text):
    return " ".join(sorted(t for t in _norm(text).split() if t not in HONORIFICS))


def norm_company(text):
    return " ".join(t for t in _norm(text).split() if t not in COMPANY_SUFFIXES)


def split_emails(text):
    return re.findall(r"[^\s;,]+@[^\s;,]+", (text or "").casefold())


def identity(name, name_en, company, company_en, emails):
    return {
        "names": {n for n in (norm_name(name), norm_name(name_en)) if n},
        "companies": {c for c in (norm_company(company), norm_company(company_en)) if c},
        "emails": set(emails),
    }


def card_identity(card):
    return identity(
        card.get("contact_person"), card.get("contact_person_english"),
        card.get("business_name"), card.get("business_name_english"),
        split_emails(" ".join(card.get("emails") or [])),
    )


def row_identity(row):
    return identity(
        cell(row, "Contact Person"), cell(row, "Contact Person (English)"),
        cell(row, "Business Name"), cell(row, "Business Name (English)"),
        split_emails(cell(row, "Email")),
    )


def find_duplicate(rows, new):
    """Return (sheet_row_number, reason) for the first matching existing row, else None."""
    for row_number, row in enumerate(rows[1:], start=2):
        old = row_identity(row)
        name_hit = new["names"] & old["names"]
        if new["emails"] & old["emails"]:
            if new["names"] and old["names"] and not name_hit:
                continue  # shared mailbox, different people
            return row_number, "email"
        if name_hit and (new["companies"] & old["companies"]):
            return row_number, "name + company"
    return None


# --------------------------------------------------------------------------
# vCard (.vcf) — works on iPhone, Android, Mac, Outlook
# --------------------------------------------------------------------------
# vCard 3.0 + UTF-8 + CRLF line endings + lines folded at 75 bytes. That is the
# combination iOS Contacts and Android Contacts (Google / Samsung) both accept.

def vc_escape(value):
    return (str(value).replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n")
            .replace(";", "\\;").replace(",", "\\,"))


def vc_fold(line):
    """Fold a line so no physical line exceeds 75 bytes (never splitting a character)."""
    if len(line.encode("utf-8")) <= 75:
        return line
    pieces, current, size = [], "", 0
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > 75:
            pieces.append(current)
            current, size = " " + ch, 1 + n  # continuation lines start with one space
        else:
            current += ch
            size += n
    pieces.append(current)
    return "\r\n".join(pieces)


def is_latin(text):
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(ch, "") for ch in letters)


def split_name(full):
    """-> (family, given, prefix). Only splits Latin-script names; others stay whole
    (splitting Arabic / Chinese / etc. names by spaces would often be wrong)."""
    parts = (full or "").split()
    prefix = []
    while len(parts) > 1 and parts[0].strip(".").casefold() in HONORIFICS:
        prefix.append(parts.pop(0))
    name = " ".join(parts)
    if len(parts) >= 2 and is_latin(name):
        return parts[-1], " ".join(parts[:-1]), " ".join(prefix)
    return "", name, " ".join(prefix)


TEL_TYPES = {
    "mobile": "CELL", "whatsapp": "CELL", "office": "WORK,VOICE",
    "toll_free": "WORK,VOICE", "fax": "WORK,FAX",
}


def clean_url(url):
    url = re.sub(r"\s+", "", url or "")
    if not url:
        return ""
    return url if "://" in url else "https://" + url


def build_vcard(card, phone_entries):
    """Return the contact as UTF-8 bytes (a .vcf file)."""
    name = (card.get("contact_person") or "").strip()
    name_en = (card.get("contact_person_english") or "").strip()
    company = (card.get("business_name") or "").strip()
    company_en = (card.get("business_name_english") or "").strip()
    title = (card.get("job_title") or "").strip()
    emails = [e.strip() for e in card.get("emails") or [] if e.strip()]

    lines = ["BEGIN:VCARD", "VERSION:3.0"]

    if name:
        family, given, prefix = split_name(name)
        lines.append(f"N:{vc_escape(family)};{vc_escape(given)};;{vc_escape(prefix)};")
        lines.append(f"FN:{vc_escape(name)}")
        if name_en and norm_name(name_en) != norm_name(name):
            lines.append(f"NICKNAME:{vc_escape(name_en)}")  # makes the contact searchable in English
    else:
        display = (company or company_en or (emails[0] if emails else "")
                   or (phone_entries[0]["pretty"] if phone_entries else "") or "Business card contact")
        lines.append(f"N:;{vc_escape(display)};;;")
        lines.append(f"FN:{vc_escape(display)}")
        if company or company_en:
            lines.append("X-ABShowAs:COMPANY")  # iPhone: show it as a company entry

    if company:
        lines.append(f"ORG:{vc_escape(company)}")
    elif company_en:
        lines.append(f"ORG:{vc_escape(company_en)}")
    if title:
        lines.append(f"TITLE:{vc_escape(title)}")

    for e in phone_entries:
        types = TEL_TYPES.get(e["label"], "WORK")
        lines.append(f"TEL;TYPE={types}:{e['dial']}")

    for email in emails:
        lines.append(f"EMAIL;TYPE=INTERNET,WORK:{email}")

    for site in [clean_url(w) for w in card.get("websites") or []][:3]:
        if site:
            lines.append(f"URL:{site}")

    address = (card.get("address") or "").strip()
    city = (card.get("city") or "").strip()
    country = (card.get("country") or "").strip()
    if address or city or country:
        country_field = country if country and country.casefold() not in address.casefold() else ""
        city_field = city if not address else ""
        lines.append(f"ADR;TYPE=WORK:;;{vc_escape(address)};{vc_escape(city_field)};;;{vc_escape(country_field)}")

    note = []
    if company_en and norm_company(company_en) != norm_company(company):
        note.append(f"Business (English): {company_en}")
    if (card.get("miscellaneous") or "").strip():
        note.append(card["miscellaneous"].strip())
    note.append("Scanned: " + datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    lines.append(f"NOTE:{vc_escape(chr(10).join(note))}")

    lines.append("END:VCARD")
    return ("\r\n".join(vc_fold(l) for l in lines) + "\r\n").encode("utf-8")


def vcard_filename(card):
    base = (card.get("contact_person_english") or card.get("contact_person")
            or card.get("business_name_english") or card.get("business_name") or "contact")
    ascii_base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    ascii_base = re.sub(r"[^A-Za-z0-9]+", "-", ascii_base).strip("-")
    return (ascii_base[:40] or "contact") + ".vcf"


def vcard_caption(card):
    who = card.get("contact_person") or card.get("business_name") or "Contact"
    extra = card.get("job_title") or ""
    company = card.get("business_name") if card.get("contact_person") else ""
    head = f"📇 {who}" + (f" — {extra}" if extra else "") + (f" @ {company}" if company else "")
    return (head + "\n\nTap the file to add to your phone:\n"
            "📱 iPhone: tap → Add Contact\n"
            "🤖 Android: tap → open with Contacts → Save")


# --------------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------------

_SEEN_UPDATES = {}  # update_id -> time, per running instance


def already_handled(update_id):
    """Telegram can deliver the same update twice. Ignore the repeat."""
    if update_id is None:
        return False
    now = time.time()
    for old in [k for k, t in _SEEN_UPDATES.items() if now - t > 900]:
        del _SEEN_UPDATES[old]
    if update_id in _SEEN_UPDATES:
        return True
    _SEEN_UPDATES[update_id] = now
    return False


def friendly_error(e):
    """Turn a technical error into a plain-English message."""
    text = str(e)
    low = text.lower()
    code = getattr(e, "code", None)
    if isinstance(e, json.JSONDecodeError):
        return ("The Google credentials saved in Vercel (GOOGLE_SERVICE_ACCOUNT_JSON) can't be read. "
                "Paste the whole key file once, from { to }, then Redeploy.")
    if "api_key_invalid" in low or "api key not valid" in low:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in Vercel (no spaces or quotes), then Redeploy."
    if code == 429 or "resource_exhausted" in low:
        return "Gemini's usage limit was reached. Wait a minute and send the card again."
    if "gemini is busy" in low or code in (500, 502, 503, 504) or "overloaded" in low or "high demand" in low:
        return "Gemini is busy right now. Please send the card again in a minute."
    if "does not have permission" in low or "permission_denied" in low:
        return "Google Sheets refused access. Share the Sheet with the service-account email as Editor."
    return f"Something went wrong: {text[:600]}"


def allowed_user_ids():
    return {int(x) for x in re.split(r"[,\s]+", env("ALLOWED_USER_IDS", "")) if x.strip().isdigit()}


def sheet_link(ws, row_number):
    sheet_id = env("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        return ""
    gid = getattr(ws, "id", None)
    anchor = f"#gid={gid}&range=A{row_number}" if gid is not None else ""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit{anchor}"


def summary_text(card, phones_text, misc_text, saved_ok, row_number=None, link="", error=None):
    if saved_ok:
        lines = [f"✅ Saved to your sheet (row {row_number})"]
    else:
        lines = [f"⚠️ Read the card but could NOT save it: {error}"]
    if card.get("business_name"):
        english = card.get("business_name_english")
        lines.append(f"🏢 {card['business_name']}" + (f" ({english})" if english else ""))
    if card.get("contact_person"):
        title = card.get("job_title")
        lines.append(f"👤 {card['contact_person']}" + (f" — {title}" if title else ""))
    if phones_text:
        lines.append(f"📞 {phones_text}")
    if card.get("emails"):
        lines.append("✉️ " + ", ".join(card["emails"]))
    place = ", ".join(p for p in (card.get("city"), card.get("country")) if p)
    if place:
        lines.append(f"📍 {place}")
    if misc_text:
        lines.append(f"📝 Misc: {misc_text[:400]}")
    if saved_ok and link:
        lines.append(f"\n📊 {link}")
    return "\n".join(lines)


def process_update(update):
    started = time.monotonic()
    msg = update.get("message")
    if not msg:
        return
    if already_handled(update.get("update_id")):
        log("ignoring repeated delivery of update", update.get("update_id"))
        return
    chat_id = msg["chat"]["id"]
    msg_id = msg["message_id"]
    user = msg.get("from") or {}
    user_id = user.get("id")

    # Private bot: only people listed in ALLOWED_USER_IDS may use it.
    if user_id not in allowed_user_ids():
        send_message(
            chat_id,
            f"🔒 This is a private bot.\nYour Telegram ID is {user_id}.\n"
            "The owner must add it to ALLOWED_USER_IDS.",
        )
        return

    file_id, unique_id = pick_image(msg)
    if not file_id:
        send_message(chat_id, HELP_TEXT)
        return

    try:
        tg_api("sendChatAction", chat_id=chat_id, action="typing")
    except Exception:
        pass

    force = "force" in (msg.get("caption") or "").casefold()

    ws, rows = open_sheet()  # fails early (before spending AI credit) if Google access is wrong

    # Cheap check first: the exact same photo was already processed.
    if not force and unique_id in {cell(r, "Telegram File ID") for r in rows[1:]}:
        send_message(chat_id, "ℹ️ This exact photo is already in the sheet, so I skipped it.", msg_id)
        return

    try:
        image_bytes = prepare_image(download_telegram_file(file_id))
    except Exception as e:
        log("image problem:", e)
        send_message(
            chat_id,
            "❌ I couldn't open that image. If it's a HEIC file, send it as a normal photo instead "
            "(not as a file).",
            msg_id,
        )
        return

    card = extract_card(image_bytes, deadline=started + REQUEST_BUDGET_SECONDS)
    if not card.get("is_business_card", True):
        send_message(chat_id, "🤔 That doesn't look like a business card, so I didn't save it. "
                              + (card.get("miscellaneous") or ""), msg_id)
        return

    # Reading the card can take a while, and the same message may have been processed by another copy
    # of this function in the meantime. Re-read the sheet NOW so anything saved meanwhile is noticed.
    # (This is what prevents double rows.)
    rows = ws.get_all_values()
    if not force and unique_id in {cell(r, "Telegram File ID") for r in rows[1:]}:
        send_message(chat_id, "ℹ️ This photo was just saved, so I didn't add it again.", msg_id)
        return

    # Person-level duplicate check (email, or name + company).
    if not force:
        duplicate = find_duplicate(rows, card_identity(card))
        if duplicate:
            row_number, reason = duplicate
            existing = rows[row_number - 1]
            who = cell(existing, "Contact Person") or cell(existing, "Contact Person (English)") or "(no name)"
            company = cell(existing, "Business Name") or cell(existing, "Business Name (English)")
            send_message(
                chat_id,
                f"♻️ Already in your sheet (row {row_number}, matched by {reason}):\n"
                f"{who}" + (f" — {company}" if company else "") + "\n\n"
                "Nothing was added. To save it anyway, send the photo again with the caption: force",
                msg_id,
            )
            return

    entries, problems = normalize_phones(
        card.get("phones"), card.get("country_iso2"), env("DEFAULT_PHONE_REGION", "IN")
    )
    phones_text = phones_cell(entries)
    misc_parts = [p for p in [card.get("miscellaneous", "").strip()] + problems if p]
    misc_text = " | ".join(misc_parts)

    added_by = f"@{user['username']}" if user.get("username") else (user.get("first_name") or str(user_id))
    row = build_row(card, phones_text, misc_text, added_by, unique_id)

    try:
        # RAW = store text exactly as given (otherwise Sheets can mangle "+91 98765..." as a formula)
        row_number = append_and_locate(ws, row, len(rows) + 1)
        send_message(
            chat_id,
            summary_text(card, phones_text, misc_text, True, row_number, sheet_link(ws, row_number)),
            msg_id,
        )
    except Exception as e:
        log("sheet write failed:", traceback.format_exc())
        send_message(chat_id, summary_text(card, phones_text, misc_text, False, error=str(e)[:200]), msg_id)

    # The vCard goes out even if the sheet write failed, so the contact is never lost.
    try:
        send_document(chat_id, build_vcard(card, entries), vcard_filename(card), vcard_caption(card))
    except Exception as e:
        log("vcard failed:", traceback.format_exc())
        send_message(chat_id, f"(I couldn't create the contact file: {str(e)[:200]})")


# --------------------------------------------------------------------------
# Vercel entry point
# --------------------------------------------------------------------------

def app(environ, start_response):
    """WSGI entry point. Vercel's Python runtime looks for a variable called `app` in app.py.
    It answers on ANY path (so https://your-domain/ and /api/webhook both work)."""

    def respond(code, body="ok"):
        data = body.encode("utf-8")
        start_response(
            f"{code} {HTTPStatus(code).phrase}",
            [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(data)))],
        )
        return [data]

    method = environ.get("REQUEST_METHOD", "GET").upper()
    if method in ("GET", "HEAD"):
        return respond(200, "Business card bot is running ✅")
    if method != "POST":
        return respond(405, "method not allowed")

    secret = env("TELEGRAM_WEBHOOK_SECRET")
    if not secret:
        return respond(500, "TELEGRAM_WEBHOOK_SECRET is not set")
    sent = environ.get("HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN", "")
    if not hmac.compare_digest(sent.encode("utf-8"), secret.encode("utf-8")):
        return respond(403, "forbidden")

    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        update = json.loads(environ["wsgi.input"].read(length) or b"{}")
    except Exception:
        return respond(400, "bad request")

    try:
        process_update(update)
    except Exception as e:
        log("ERROR:", traceback.format_exc())
        try:
            chat_id = update["message"]["chat"]["id"]
            send_message(chat_id, f"❌ {friendly_error(e)}")
        except Exception:
            pass

    # Always answer 200 so Telegram doesn't re-send the same update again and again.
    return respond(200, "ok")
