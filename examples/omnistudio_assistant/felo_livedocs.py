"""Owner-bound, Firebase-authenticated proxy for the Felo LiveDoc API.

LiveDoc references are signed by the gateway, so ownership survives restarts
without a shared, unscoped in-memory ID table. Raw Felo LiveDoc IDs are never
accepted from a caller. Every operation still requires a verified Firebase UID.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from email import policy
from email.parser import BytesParser
from typing import Any, Callable

BASE = "/api/felo/livedocs"
API_ROOT = "https://openapi.felo.ai"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_OVERHEAD = 64 * 1024
RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SHORT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _uid_tag(uid: str, secret: str) -> str:
    return hmac.new(secret.encode(), b"felo-livedoc-owner-v1\0" + uid.encode(), hashlib.sha256).hexdigest()


def make_ref(short_id: str, uid: str, secret: str) -> str:
    if not SHORT_ID_RE.fullmatch(short_id):
        raise ValueError("invalid_short_id")
    payload = json.dumps({"v": 1, "id": short_id, "owner": _uid_tag(uid, secret)}, separators=(",", ":")).encode()
    encoded = _b64url(payload)
    signature = hmac.new(secret.encode(), b"felo-livedoc-ref-v1\0" + encoded.encode(), hashlib.sha256).digest()
    return encoded + "." + _b64url(signature)


def read_ref(ref: str, uid: str, secret: str) -> str | None:
    if not isinstance(ref, str) or len(ref) > 2048:
        return None
    try:
        encoded, sig_text = ref.split(".", 1)
        supplied = _unb64url(sig_text)
        expected = hmac.new(secret.encode(), b"felo-livedoc-ref-v1\0" + encoded.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            return None
        payload = json.loads(_unb64url(encoded))
        expected_owner = _uid_tag(uid, secret)
        if not isinstance(payload, dict) or payload.get("v") != 1 or not isinstance(payload.get("owner"), str) or not hmac.compare_digest(payload["owner"], expected_owner):
            return None
        short_id = payload.get("id")
        return short_id if isinstance(short_id, str) and SHORT_ID_RE.fullmatch(short_id) else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in {"short_id", "live_doc_short_id", "livedoc_short_id"}}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _body_bytes(handler: Any, max_bytes: int) -> bytes | None:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except (TypeError, ValueError):
        handler._send_json(400, {"error": "invalid_content_length"})
        return None
    if "chunked" in handler.headers.get("Transfer-Encoding", "").lower():
        handler._send_json(400, {"error": "unsupported_transfer_encoding"})
        return None
    if length <= 0:
        handler._send_json(400, {"error": "empty_request"})
        return None
    if length > max_bytes:
        handler._send_json(413, {"error": "request_too_large"})
        return None
    raw = handler.rfile.read(length)
    if len(raw) != length:
        handler._send_json(400, {"error": "incomplete_request"})
        return None
    return raw


def _json_body(handler: Any) -> dict[str, Any] | None:
    raw = _body_bytes(handler, MAX_JSON_BYTES)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        handler._send_json(400, {"error": "invalid_json"})
        return None
    if not isinstance(value, dict):
        handler._send_json(400, {"error": "json_object_required"})
        return None
    return value


def _multipart_file(handler: Any) -> tuple[bytes, str, str, str | None] | None:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data;"):
        handler._send_json(400, {"error": "multipart_form_data_required"})
        return None
    raw = _body_bytes(handler, MAX_UPLOAD_BYTES + MAX_UPLOAD_OVERHEAD)
    if raw is None:
        return None
    try:
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii", "strict") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
        )
    except (UnicodeEncodeError, ValueError):
        handler._send_json(400, {"error": "invalid_multipart"})
        return None
    if not message.is_multipart():
        handler._send_json(400, {"error": "invalid_multipart"})
        return None
    file_parts: list[tuple[bytes, str, str]] = []
    title: str | None = None
    for part in message.iter_parts():
        field = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if field == "file":
            filename = part.get_filename() or "upload.bin"
            filename = os.path.basename(filename.replace("\\", "/"))
            filename = re.sub(r"[\x00-\x1f\x7f]", "", filename).replace('"', "_")[:200] or "upload.bin"
            media_type = part.get_content_type()
            if not re.fullmatch(r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", media_type):
                media_type = "application/octet-stream"
            file_parts.append((data, filename, media_type))
        elif field == "title":
            if title is not None:
                handler._send_json(400, {"error": "duplicate_title"})
                return None
            try:
                title = data.decode(part.get_content_charset() or "utf-8", "strict").strip()
            except (LookupError, UnicodeDecodeError):
                handler._send_json(400, {"error": "invalid_title_encoding"})
                return None
        else:
            handler._send_json(400, {"error": "unsupported_multipart_field"})
            return None
    if len(file_parts) != 1:
        handler._send_json(400, {"error": "exactly_one_file_required"})
        return None
    data, filename, media_type = file_parts[0]
    if not data:
        handler._send_json(400, {"error": "empty_file"})
        return None
    if len(data) > MAX_UPLOAD_BYTES:
        handler._send_json(413, {"error": "file_too_large"})
        return None
    if title is not None and len(title) > 500:
        handler._send_json(400, {"error": "title_too_long"})
        return None
    return data, filename, media_type, title


def _felo_upload(path: str, file_data: bytes, filename: str, media_type: str, title: str | None,
                 api_key: str, error_cls: type[Exception]) -> dict[str, Any]:
    if not api_key:
        raise error_cls("felo_not_configured", 503)
    boundary = "----ZapiaFelo" + secrets.token_hex(16)
    chunks = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {media_type}\r\n\r\n".encode("utf-8"),
        file_data,
        b"\r\n",
    ]
    if title is not None:
        safe_title = title.replace("\r", " ").replace("\n", " ")
        chunks.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\n{safe_title}\r\n".encode("utf-8")])
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    request = urllib.request.Request(
        API_ROOT + path,
        data=b"".join(chunks),
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    class _NoUploadRedirect(urllib.request.HTTPRedirectHandler):
        # Never forward the server-side Felo bearer token to a redirect target.
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        response = urllib.request.build_opener(_NoUploadRedirect()).open(request, timeout=90)
    except urllib.error.HTTPError as exc:
        retry = exc.headers.get("Retry-After") if exc.headers else None
        if exc.code == 429:
            raise error_cls("felo_rate_limited", 429, retry) from None
        if exc.code == 402:
            raise error_cls("felo_insufficient_credits", 402) from None
        if exc.code in (401, 403):
            raise error_cls("felo_auth_failed", 502) from None
        raise error_cls("felo_provider_error", 502) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise error_cls("felo_unavailable", 502) from None
    with response:
        raw = response.read(512 * 1024 + 1)
        response_type = response.headers.get("Content-Type", "")
    if len(raw) > 512 * 1024 or "json" not in response_type.lower():
        raise error_cls("felo_invalid_response", 502)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise error_cls("felo_invalid_response", 502) from None
    if not isinstance(payload, dict) or payload.get("status") not in ("ok", 200, "200"):
        raise error_cls("felo_provider_error", 502)
    return payload


def _download_resource(handler: Any, short_id: str, resource_id: str, expires_in: str | None,
                       api_key: str, error_cls: type[Exception], send_error: Callable[..., None]) -> None:
    if not api_key:
        raise error_cls("felo_not_configured", 503)
    path = f"/v2/livedocs/{urllib.parse.quote(short_id, safe='')}/resources/{urllib.parse.quote(resource_id, safe='')}/download"
    query = urllib.parse.urlencode({"expires_in": expires_in}) if expires_in else ""
    request = urllib.request.Request(
        API_ROOT + path + ("?" + query if query else ""),
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/octet-stream"}, method="GET"
    )
    class _SafeFeloRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            destination = urllib.parse.urlsplit(newurl)
            host = (destination.hostname or "").lower()
            allowed_host = host == "openapi.felo.ai" or host.endswith(".amazonaws.com") or host.endswith(".amazonaws.com.cn")
            if destination.scheme != "https" or not allowed_host:
                raise urllib.error.URLError("blocked redirect")
            # Signed download URLs are self-authorizing; do not copy Felo's
            # bearer token (or other request headers) to the storage host.
            return urllib.request.Request(newurl, headers={"Accept": "application/octet-stream"}, method="GET")

    try:
        response = urllib.request.build_opener(_SafeFeloRedirect()).open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        retry = exc.headers.get("Retry-After") if exc.headers else None
        if exc.code == 429:
            raise error_cls("felo_rate_limited", 429, retry) from None
        if exc.code == 402:
            raise error_cls("felo_insufficient_credits", 402) from None
        raise error_cls("felo_provider_error", 502) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise error_cls("felo_unavailable", 502) from None
    with response:
        raw = response.read(MAX_UPLOAD_BYTES + 1)
        media_type = response.headers.get("Content-Type", "application/octet-stream")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise error_cls("felo_download_too_large", 413)
    if not re.fullmatch(r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", media_type.split(";", 1)[0].strip()):
        media_type = "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", media_type)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Content-Disposition", f'attachment; filename="livedoc-resource-{resource_id}.bin"')
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(raw)


def _valid_page_query(parsed: Any, allowed: set[str], handler: Any) -> str | None:
    values = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
    if set(values) - allowed:
        handler._send_json(400, {"error": "unsupported_query_parameter"})
        return None
    clean: list[tuple[str, str]] = []
    for key, entries in values.items():
        if key in {"page", "size", "status", "expires_in"}:
            if len(entries) != 1 or not entries[0].isdigit():
                handler._send_json(400, {"error": f"invalid_{key}"})
                return None
            number = int(entries[0])
            if key == "status" and number not in {0, 1, 2}:
                handler._send_json(400, {"error": "invalid_status"})
                return None
            maximum = 100 if key == "size" else (86400 if key == "expires_in" else 1000000)
            minimum = 1 if key in {"page", "size", "expires_in"} else 0
            if number < minimum or number > maximum:
                handler._send_json(400, {"error": f"invalid_{key}"})
                return None
            clean.append((key, str(number)))
        elif key == "record_type":
            if len(entries) != 1 or entries[0] not in {"comment", "edit", "status_change"}:
                handler._send_json(400, {"error": "invalid_record_type"})
                return None
            clean.append((key, entries[0]))
        elif key == "keyword":
            if len(entries) != 1 or len(entries[0]) > 200:
                handler._send_json(400, {"error": "invalid_keyword"})
                return None
            clean.append((key, entries[0]))
        elif key == "resource_types":
            if len(entries) > 20 or any(len(item) > 40 for item in entries):
                handler._send_json(400, {"error": "invalid_resource_types"})
                return None
            clean.extend((key, item) for item in entries)
        elif key == "labels":
            if len(entries) > 10 or any(len(item) > 50 for item in entries):
                handler._send_json(400, {"error": "invalid_labels"})
                return None
            clean.extend((key, item) for item in entries)
    return urllib.parse.urlencode(clean, doseq=True)


def _string_fields(body: dict[str, Any], allowed: set[str], limits: dict[str, int], *, required: set[str] = frozenset()) -> bool:
    if set(body) - allowed or required - set(body):
        return False
    for key, value in body.items():
        if not isinstance(value, str) or len(value) > limits.get(key, 1000000):
            return False
    return True


def _task_body(body: dict[str, Any], *, create: bool) -> bool:
    allowed = {"title", "description", "status", "sort", "labels", "operated_by"}
    required = {"title", "status", "sort"} if create else set()
    if set(body) - allowed or required - set(body):
        return False
    for name in ("title", "description", "operated_by"):
        if name in body and (not isinstance(body[name], str) or len(body[name]) > (500 if name == "title" else 20000 if name == "description" else 100)):
            return False
    for name in ("status", "sort"):
        if name in body:
            value = body[name]
            if isinstance(value, bool) or not isinstance(value, int) or (name == "status" and value not in {0, 1, 2}) or (name == "sort" and value < 0):
                return False
    if "labels" in body:
        labels = body["labels"]
        if not isinstance(labels, list) or len(labels) > 10 or any(not isinstance(x, str) or len(x) > 50 for x in labels):
            return False
    return bool(body) or not create


def _send_payload(handler: Any, status: int, payload: dict[str, Any]) -> None:
    handler._send_json(status, _clean(payload))


def handle(handler: Any, uid: str, method: str, parsed: Any,
           felo_request: Callable[..., tuple[dict[str, Any], Any]],
           error_cls: type[Exception], send_error: Callable[..., None],
           owner_secret: str, api_key: str) -> bool:
    """Handle one /api/felo/livedocs request. Return False for other routes."""
    route = parsed.path
    if route != BASE and not route.startswith(BASE + "/"):
        return False
    try:
        if method == "GET" and route == BASE:
            handler._send_json(405, {"error": "use_post_list_with_owned_doc_refs"})
            return True
        if method == "POST" and route == BASE:
            body = _json_body(handler)
            if body is None:
                return True
            if not _string_fields(body, {"name", "description", "icon"}, {"name": 200, "description": 2000, "icon": 500}, required={"name"}) or not body.get("name", "").strip():
                handler._send_json(400, {"error": "valid_name_required"})
                return True
            upstream, _ = felo_request("POST", "/v2/livedocs", body)
            data = upstream.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("short_id"), str):
                raise error_cls("felo_invalid_response", 502)
            doc_ref = make_ref(data["short_id"], uid, owner_secret)
            safe_data = {k: _clean(v) for k, v in data.items() if k != "short_id"}
            safe_data["doc_ref"] = doc_ref
            handler._send_json(200, {"status": upstream.get("status"), "data": safe_data})
            return True
        if method == "POST" and route == BASE + "/list":
            body = _json_body(handler)
            if body is None:
                return True
            refs = body.get("doc_refs")
            if not isinstance(refs, list) or not refs or len(refs) > 100:
                handler._send_json(400, {"error": "owned_doc_refs_required_max_100"})
                return True
            owned: dict[str, str] = {}
            for ref in refs:
                doc_id = read_ref(ref, uid, owner_secret)
                if doc_id:
                    owned[doc_id] = ref
            if not owned:
                handler._send_json(404, {"error": "livedocs_not_found"})
                return True
            page = body.get("page", 1); size = body.get("size", 50); keyword = body.get("keyword")
            if isinstance(page, bool) or not isinstance(page, int) or page < 1 or isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 100 or (keyword is not None and (not isinstance(keyword, str) or len(keyword) > 200)):
                handler._send_json(400, {"error": "invalid_list_parameters"})
                return True
            query = {"page": page, "size": size}
            if keyword: query["keyword"] = keyword
            upstream, _ = felo_request("GET", "/v2/livedocs?" + urllib.parse.urlencode(query))
            data = upstream.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise error_cls("felo_invalid_response", 502)
            items = []
            for item in data["items"]:
                if isinstance(item, dict) and item.get("short_id") in owned:
                    value = {k: _clean(v) for k, v in item.items() if k != "short_id"}
                    value["doc_ref"] = owned[item["short_id"]]
                    items.append(value)
            handler._send_json(200, {"status": upstream.get("status"), "data": {"page": page, "size": size, "items": items, "owned_total_on_page": len(items)}})
            return True
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            handler._send_json(405, {"error": "method_not_allowed"})
            return True
        ref = handler.headers.get("X-Felo-LiveDoc-Ref", "")
        short_id = read_ref(ref, uid, owner_secret)
        if not short_id:
            handler._send_json(404, {"error": "livedoc_not_found"})
            return True
        qbase = f"/v2/livedocs/{urllib.parse.quote(short_id, safe='')}"
        suffix = route[len(BASE):]
        resource_base = BASE + "/resources"
        task_base = BASE + "/tasks"
        upstream_path: str | None = None
        body: dict[str, Any] | None = None
        if method == "GET":
            if route == resource_base:
                query = _valid_page_query(parsed, {"resource_types", "page", "size"}, handler)
                if query is None: return True
                upstream_path = qbase + "/resources" + ("?" + query if query else "")
            elif route == BASE + "/readme":
                upstream_path = qbase + "/readme"
            elif route == BASE + "/tasks":
                query = _valid_page_query(parsed, {"status", "labels", "page", "size"}, handler)
                if query is None: return True
                upstream_path = qbase + "/tasks" + ("?" + query if query else "")
            else:
                match = re.fullmatch(re.escape(resource_base) + r"/([A-Za-z0-9_-]{1,128})(?:/(content|download))?", route)
                task_match = re.fullmatch(re.escape(task_base) + r"/([A-Za-z0-9_-]{1,128})/records", route)
                if match:
                    resource_id, action = match.groups()
                    upstream_path = qbase + "/resources/" + urllib.parse.quote(resource_id, safe="")
                    if action == "content": upstream_path += "/content"
                    if action == "download":
                        query = _valid_page_query(parsed, {"expires_in"}, handler)
                        if query is None: return True
                        expires = urllib.parse.parse_qs(query).get("expires_in", [None])[0]
                        _download_resource(handler, short_id, resource_id, expires, api_key, error_cls, send_error)
                        return True
                elif task_match:
                    task_id = task_match.group(1)
                    query = _valid_page_query(parsed, {"record_type", "page", "size"}, handler)
                    if query is None: return True
                    upstream_path = qbase + f"/tasks/{urllib.parse.quote(task_id, safe='')}/records" + ("?" + query if query else "")
        elif method == "POST":
            if route == resource_base + "/doc":
                body = _json_body(handler)
                if body is None: return True
                if not _string_fields(body, {"title", "content"}, {"title": 500, "content": 1000000}, required={"content"}):
                    handler._send_json(400, {"error": "valid_document_content_required"}); return True
                upstream_path = qbase + "/resources/doc"
            elif route in {resource_base + "/upload", resource_base + "/upload-doc"}:
                parsed_file = _multipart_file(handler)
                if parsed_file is None: return True
                file_data, filename, media_type, title = parsed_file
                if route.endswith("/upload") and title is not None:
                    handler._send_json(400, {"error": "title_not_supported_for_file_resource"}); return True
                path = qbase + ("/resources/upload-doc" if route.endswith("upload-doc") else "/resources/upload")
                uploaded = _felo_upload(path, file_data, filename, media_type, title, api_key, error_cls)
                handler._send_json(200, {"status": uploaded.get("status"), "data": _clean(uploaded.get("data"))})
                return True
            elif route == resource_base + "/urls":
                body = _json_body(handler)
                if body is None: return True
                urls = body.get("urls")
                if set(body) != {"urls"} or not isinstance(urls, list) or not 1 <= len(urls) <= 10:
                    handler._send_json(400, {"error": "one_to_ten_urls_required"}); return True
                clean_urls = []
                for item in urls:
                    value = {"url": item} if isinstance(item, str) else item
                    if not isinstance(value, dict) or set(value) - {"url", "title"} or not isinstance(value.get("url"), str) or not 5 <= len(value["url"]) <= 2000:
                        handler._send_json(400, {"error": "invalid_url_item"}); return True
                    parsed_url = urllib.parse.urlsplit(value["url"])
                    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or parsed_url.username or parsed_url.password:
                        handler._send_json(400, {"error": "http_url_required"}); return True
                    if "title" in value and (not isinstance(value["title"], str) or len(value["title"]) > 1000):
                        handler._send_json(400, {"error": "invalid_url_title"}); return True
                    clean_urls.append(value)
                body = {"urls": clean_urls}; upstream_path = qbase + "/resources/urls"
            elif route in {resource_base + "/retrieve", resource_base + "/route", resource_base + "/ppt-retrieve"}:
                body = _json_body(handler)
                if body is None: return True
                if route.endswith("/retrieve"):
                    valid = isinstance(body.get("query"), str) and 1 <= len(body["query"]) <= 4000 and set(body) <= {"query", "resource_ids"}
                    ids = body.get("resource_ids")
                    valid = valid and (ids is None or (isinstance(ids, list) and 1 <= len(ids) <= 50 and all(isinstance(x, str) and RESOURCE_ID_RE.fullmatch(x) for x in ids)))
                elif route.endswith("/route"):
                    count = body.get("max_resources", 5)
                    valid = isinstance(body.get("query"), str) and 1 <= len(body["query"]) <= 4000 and set(body) <= {"query", "max_resources"} and isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= 50
                else:
                    page = body.get("page_number"); count = body.get("max_chunk", 3)
                    valid = set(body) <= {"resource_id", "page_number", "query", "max_chunk"} and isinstance(body.get("resource_id"), str) and RESOURCE_ID_RE.fullmatch(body["resource_id"]) and isinstance(page, int) and not isinstance(page, bool) and page >= 1 and isinstance(body.get("query"), str) and 1 <= len(body["query"]) <= 4000 and isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= 20 and {"resource_id", "page_number", "query"} <= set(body)
                if not valid:
                    handler._send_json(400, {"error": "invalid_retrieval_request"}); return True
                upstream_path = qbase + ("/resources/retrieve" if route.endswith("/retrieve") else "/resources/route" if route.endswith("/route") else "/resources/ppt-retrieve")
            elif route == BASE + "/readme/append":
                body = _json_body(handler)
                if body is None: return True
                if not _string_fields(body, {"content"}, {"content": 1000000}, required={"content"}) or not body["content"]:
                    handler._send_json(400, {"error": "readme_content_required"}); return True
                upstream_path = qbase + "/readme/append"
            elif route == BASE + "/tasks":
                body = _json_body(handler)
                if body is None: return True
                if not _task_body(body, create=True):
                    handler._send_json(400, {"error": "invalid_task"}); return True
                upstream_path = qbase + "/tasks"
            else:
                task_comment = re.fullmatch(re.escape(task_base) + r"/([A-Za-z0-9_-]{1,128})/comments", route)
                if task_comment:
                    body = _json_body(handler)
                    if body is None: return True
                    if not _string_fields(body, {"content", "operated_by"}, {"content": 20000, "operated_by": 100}, required={"content"}) or not body["content"]:
                        handler._send_json(400, {"error": "comment_content_required"}); return True
                    upstream_path = qbase + f"/tasks/{urllib.parse.quote(task_comment.group(1), safe='')}/comments"
        elif method == "PUT":
            if route == BASE:
                body = _json_body(handler)
                if body is None: return True
                if not _string_fields(body, {"name", "description", "icon"}, {"name": 200, "description": 2000, "icon": 500}) or not body:
                    handler._send_json(400, {"error": "invalid_livedoc_update"}); return True
                upstream_path = qbase
            elif route == BASE + "/readme":
                body = _json_body(handler)
                if body is None: return True
                if not _string_fields(body, {"summary", "content"}, {"summary": 2000, "content": 1000000}) or not body:
                    handler._send_json(400, {"error": "invalid_readme_update"}); return True
                upstream_path = qbase + "/readme"
            else:
                match = re.fullmatch(re.escape(resource_base) + r"/([A-Za-z0-9_-]{1,128})(?:/(content))?", route)
                if match:
                    resource_id, action = match.groups()
                    body = _json_body(handler)
                    if body is None: return True
                    if action == "content":
                        valid = _string_fields(body, {"content"}, {"content": 1000000}, required={"content"})
                        tail = "/content"
                    else:
                        valid = _string_fields(body, {"title", "snippet", "thumbnail"}, {"title": 500, "snippet": 2000, "thumbnail": 2000}) and bool(body)
                        tail = ""
                    if not valid:
                        handler._send_json(400, {"error": "invalid_resource_update"}); return True
                    upstream_path = qbase + f"/resources/{urllib.parse.quote(resource_id, safe='')}" + tail
        elif method == "PATCH":
            match = re.fullmatch(re.escape(task_base) + r"/([A-Za-z0-9_-]{1,128})", route)
            if match:
                body = _json_body(handler)
                if body is None: return True
                if not _task_body(body, create=False):
                    handler._send_json(400, {"error": "invalid_task_update"}); return True
                upstream_path = qbase + f"/tasks/{urllib.parse.quote(match.group(1), safe='')}"
        elif method == "DELETE":
            if route == BASE:
                upstream_path = qbase
            elif route == BASE + "/readme":
                upstream_path = qbase + "/readme"
            else:
                resource_match = re.fullmatch(re.escape(resource_base) + r"/([A-Za-z0-9_-]{1,128})", route)
                task_match = re.fullmatch(re.escape(task_base) + r"/([A-Za-z0-9_-]{1,128})", route)
                if resource_match: upstream_path = qbase + f"/resources/{urllib.parse.quote(resource_match.group(1), safe='')}"
                elif task_match: upstream_path = qbase + f"/tasks/{urllib.parse.quote(task_match.group(1), safe='')}"
        if upstream_path is None:
            handler._send_json(404, {"error": "not_found"})
            return True
        payload, _ = felo_request(method, upstream_path, body)
        if not isinstance(payload, dict):
            raise error_cls("felo_invalid_response", 502)
        _send_payload(handler, 200, payload)
    except error_cls as exc:
        send_error(handler, exc)
    except (ValueError, TypeError, KeyError):
        handler._send_json(400, {"error": "invalid_livedoc_request"})
    return True
