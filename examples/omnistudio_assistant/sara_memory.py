"""Read only the signed-in user's approved Sara notes from Firebase RTDB."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_DATABASE_URL = "https://omnistudio-caaf5-default-rtdb.firebaseio.com"
MAX_RESPONSE_BYTES = 256 * 1024
MAX_NOTE_CHARS = 2000
MAX_CONTEXT_ENTRIES = 5
MAX_CONTEXT_CHARS = 5000

OVERVIEW_PHRASES = (
    "que sabes de mi", "que sabes sobre mi", "que sabes acerca de mi",
    "que recuerdas de mi", "que recuerdas sobre mi", "que informacion tienes de mi",
    "que informacion tienes sobre mi", "what do you know about me",
    "what do you remember about me", "what information do you have about me",
)
DIRECT_RECALL_PHRASES = (
    "como me llamo", "cual es mi nombre", "quien soy", "mi nombre", "mi correo",
    "mi email", "mi edad", "mi cumpleanos", "mi fecha de nacimiento", "mi direccion",
    "mi telefono", "mis preferencias", "my name", "my email", "my age", "my birthday",
    "my address", "my phone", "my preferences",
)
RECALL_WORDS = {
    "recuerda", "recuerdas", "recuerdo", "recordar", "acuerdas", "acuerdo",
    "sabes", "sabe", "saber", "conoces", "conoce", "conocer", "remember",
    "recall", "know", "knows",
}
PERSONAL_REFERENCES = (
    "sobre mi", "de mi", "about me", "remember me", "recuerdas mi", "recuerda mi",
    "sabes mi", "conoces mi", "know my", "remember my",
)
STOP_WORDS = {
    "que", "como", "cuando", "porque", "esto", "esta", "este", "desde", "hasta", "sobre",
    "para", "mi", "mis", "me", "mio", "mia", "yo", "the", "and", "for", "with", "that",
    "this", "from", "your", "have", "what", "when", "where", "about", "you", "do", "is",
    "are", "my", "of", "to", "a", "an", "in", "on", "it", "i", "am", "tell",
}
SYNONYMS = (
    (
        {"nombre", "name", "llamo", "llama", "llamarse", "llamado", "llamada", "called"},
        {"nombre", "name", "llamo", "llama", "llamarse", "llamado", "llamada", "called"},
    ),
    ({"correo", "email", "e-mail"}, {"correo", "email", "e-mail"}),
    ({"telefono", "phone", "celular", "mobile"}, {"telefono", "phone", "celular", "mobile"}),
    ({"cumpleanos", "birthday", "nacimiento", "birth"}, {"cumpleanos", "birthday", "nacimiento", "birth"}),
    ({"edad", "age"}, {"edad", "age"}),
    ({"direccion", "address"}, {"direccion", "address"}),
)


class SaraMemoryFetchError(Exception):
    """Sanitized memory-fetch failure; never carries a URL, token, or response body."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(ch for ch in decomposed if not unicodedata.combining(ch)).split())


def is_assistant_name_query(query: str) -> bool:
    normalized = normalize(query)
    return any(phrase in normalized for phrase in (
        "como te llamas", "cual es tu nombre", "tu nombre", "quien eres", "eres sara",
        "what is your name", "what's your name", "what are you called", "who are you",
    ))


def is_user_name_query(query: str) -> bool:
    normalized = normalize(query)
    return any(phrase in normalized for phrase in (
        "como me llamo", "cual es mi nombre", "quien soy", "what is my name",
        "what's my name", "what do i call myself", "how am i called",
    ))


def assistant_name_reply() -> str:
    return "Me llamo Sara, soy la asistente de OmniStudio."


def format_user_name_reply(context: str) -> str:
    notes = []
    for line in context.splitlines():
        note = line.strip()
        if note.startswith("•"):
            note = note[1:].strip()
        if note:
            notes.append(note)
    if not notes:
        return "No encuentro tu nombre en las notas que guardaste; no voy a adivinarlo."
    note = notes[0].rstrip(" .")
    return f"Según la nota que guardaste: {note}."


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[^\W_]+", normalize(text), flags=re.UNICODE)
            if len(word) >= 3 and word not in STOP_WORDS}


def is_overview_query(query: str) -> bool:
    normalized = normalize(query)
    return any(phrase in normalized for phrase in OVERVIEW_PHRASES)


def is_personal_recall_query(query: str) -> bool:
    normalized = normalize(query)
    if not normalized:
        return False
    if is_overview_query(normalized) or is_user_name_query(normalized) or any(
        phrase in normalized for phrase in DIRECT_RECALL_PHRASES
    ):
        return True
    words = set(_tokens(normalized))
    asks_to_recall = bool(words & RECALL_WORDS)
    asks_about_user = any(reference in normalized for reference in PERSONAL_REFERENCES)
    return asks_to_recall and asks_about_user


def _query_terms(query: str) -> set[str]:
    terms = _tokens(query)
    for aliases, additions in SYNONYMS:
        if terms & aliases:
            terms.update(additions)
    return terms


def select_relevant_context(entries: list[dict[str, Any]], query: str) -> str:
    valid: list[tuple[str, int]] = []
    for entry in entries:
        text = entry.get("text")
        created_at = entry.get("createdAt", 0)
        if not isinstance(text, str) or not text.strip():
            continue
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
            created_at = 0
        valid.append((text.strip()[:MAX_NOTE_CHARS], int(created_at)))
    valid.sort(key=lambda item: item[1], reverse=True)
    if not valid:
        return ""
    if is_overview_query(query):
        selected = [text for text, _ in valid[:MAX_CONTEXT_ENTRIES]]
    else:
        terms = _query_terms(query)
        ranked = [(len(_tokens(text) & terms), created_at, text) for text, created_at in valid]
        selected = [text for score, _, text in sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True) if score > 0][:MAX_CONTEXT_ENTRIES]
        if not selected and is_personal_recall_query(query):
            # A personal recall request may use wording absent from the saved note.
            # Supply a small, recent slice of this same user's approved notes;
            # the fallback prompt must answer only from relevant entries.
            selected = [text for text, _ in valid[:MAX_CONTEXT_ENTRIES]]
    return "\n".join(f"• {text}" for text in selected)[:MAX_CONTEXT_CHARS]


def fetch_relevant_context(
    database_url: str,
    uid: str,
    firebase_id_token: str,
    query: str,
    *,
    urlopen=urllib.request.urlopen,
    status_callback=None,
) -> str:
    """Fetch user's notes with the caller's ID token so RTDB rules remain enforced."""
    if not uid or not firebase_id_token:
        raise SaraMemoryFetchError("identity_unavailable")
    base = database_url.rstrip("/")
    parsed_base = urllib.parse.urlsplit(base)
    host = (parsed_base.hostname or "").lower()
    trusted_firebase_host = host.endswith(".firebaseio.com") or host.endswith(".firebasedatabase.app")
    if (
        parsed_base.scheme != "https"
        or not trusted_firebase_host
        or parsed_base.username is not None
        or parsed_base.password is not None
    ):
        raise SaraMemoryFetchError("invalid_database_url")
    path = urllib.parse.quote(uid, safe="")
    auth = urllib.parse.urlencode({"auth": firebase_id_token})
    request = urllib.request.Request(
        f"{base}/sara_knowledge/{path}.json?{auth}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=8) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise SaraMemoryFetchError(f"http_{exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise SaraMemoryFetchError("unavailable") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SaraMemoryFetchError("response_too_large")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SaraMemoryFetchError("invalid_json") from None
    if payload is None:
        if status_callback:
            status_callback({"records": 0, "dict_records": 0, "owner_match": 0,
                             "owner_missing": 0, "owner_mismatch": 0,
                             "nonempty_text": 0, "selected": 0})
        return ""
    if not isinstance(payload, dict):
        raise SaraMemoryFetchError("unexpected_shape")
    # The authenticated, UID-scoped RTDB path and security rules enforce ownership.
    # Accept legacy notes without ownerUid, but reject any explicit mismatch.
    dict_entries = [entry for entry in payload.values() if isinstance(entry, dict)]
    own_entries = [
        entry for entry in dict_entries
        if entry.get("ownerUid") in (None, "", uid)
    ]
    context = select_relevant_context(own_entries, query)
    if status_callback:
        owner_missing = sum(entry.get("ownerUid") in (None, "") for entry in dict_entries)
        owner_match = sum(entry.get("ownerUid") == uid for entry in dict_entries)
        owner_mismatch = len(dict_entries) - owner_missing - owner_match
        nonempty_text = sum(
            isinstance(entry.get("text"), str) and bool(entry.get("text", "").strip())
            for entry in own_entries
        )
        status_callback({
            "records": len(payload),
            "dict_records": len(dict_entries),
            "owner_match": owner_match,
            "owner_missing": owner_missing,
            "owner_mismatch": owner_mismatch,
            "nonempty_text": nonempty_text,
            "selected": len(context.splitlines()) if context else 0,
        })
    return context

