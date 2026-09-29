"""Mock-only LiveDoc gateway tests; no Felo calls or credits are used."""
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

os.environ.setdefault("RASA_AUTH_TOKEN", "test-only-not-a-real-secret")
try:
    import jwt  # noqa: F401
except ImportError:
    jwt_stub = types.ModuleType("jwt")
    jwt_stub.encode = lambda *args, **kwargs: "test-token"
    sys.modules["jwt"] = jwt_stub

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("secure_gateway_livedoc_test", ROOT / "secure_gateway.py")
gw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gw
spec.loader.exec_module(gw)


class FakeHandler:
    def __init__(self, body=b"", content_type="application/json", doc_ref=None):
        self.rfile = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body)), "Content-Type": content_type}
        if doc_ref:
            self.headers["X-Felo-LiveDoc-Ref"] = doc_ref
        self.responses = []
        self.sent_headers = []
        self.wfile = io.BytesIO()

    def _send_json(self, status, payload):
        self.responses.append((status, payload))

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.sent_headers.append((name, value))

    def end_headers(self):
        pass


class LiveDocTests(unittest.TestCase):
    secret = "test-only-not-a-real-secret"

    def test_signed_ref_is_owner_bound_and_tamper_resistant(self):
        ref = gw.felo_livedocs.make_ref("doc-123", "uid-a", self.secret)
        self.assertEqual(gw.felo_livedocs.read_ref(ref, "uid-a", self.secret), "doc-123")
        self.assertIsNone(gw.felo_livedocs.read_ref(ref, "uid-b", self.secret))
        self.assertIsNone(gw.felo_livedocs.read_ref(ref[:-1] + ("A" if ref[-1] != "A" else "B"), "uid-a", self.secret))

    def test_create_returns_signed_ref_without_raw_provider_id(self):
        handler = FakeHandler(json.dumps({"name": "Research doc"}).encode())
        upstream = Mock(return_value=({"status": "ok", "data": {"short_id": "doc-123", "name": "Research doc"}}, "application/json"))
        self.assertTrue(gw.felo_livedocs.handle(handler, "uid-a", "POST", urlsplit(gw.felo_livedocs.BASE), upstream, gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key"))
        status, payload = handler.responses[-1]
        self.assertEqual(status, 200)
        data = payload["data"]
        self.assertNotIn("short_id", data)
        self.assertEqual(gw.felo_livedocs.read_ref(data["doc_ref"], "uid-a", self.secret), "doc-123")

    def test_resource_access_uses_ref_and_denies_other_uid_before_upstream(self):
        ref = gw.felo_livedocs.make_ref("doc-123", "uid-a", self.secret)
        handler = FakeHandler(doc_ref=ref)
        upstream = Mock(return_value=({"status": "ok", "data": {"items": [{"id": "res-1"}]}}, "application/json"))
        gw.felo_livedocs.handle(handler, "uid-a", "GET", urlsplit(gw.felo_livedocs.BASE + "/resources?page=1"), upstream, gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key")
        upstream.assert_called_once_with("GET", "/v2/livedocs/doc-123/resources?page=1", None)
        self.assertEqual(handler.responses[-1][0], 200)

        other = FakeHandler(doc_ref=ref)
        forbidden_upstream = Mock()
        gw.felo_livedocs.handle(other, "uid-b", "GET", urlsplit(gw.felo_livedocs.BASE + "/resources"), forbidden_upstream, gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key")
        forbidden_upstream.assert_not_called()
        self.assertEqual(other.responses[-1], (404, {"error": "livedoc_not_found"}))

    def test_list_filters_provider_records_to_owned_refs(self):
        ref = gw.felo_livedocs.make_ref("owned-1", "uid-a", self.secret)
        handler = FakeHandler(json.dumps({"doc_refs": [ref], "page": 1, "size": 10}).encode())
        upstream = Mock(return_value=({"status": "ok", "data": {"items": [
            {"short_id": "owned-1", "name": "mine"},
            {"short_id": "other-1", "name": "not mine"},
        ]}}, "application/json"))
        gw.felo_livedocs.handle(handler, "uid-a", "POST", urlsplit(gw.felo_livedocs.BASE + "/list"), upstream, gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key")
        self.assertEqual([item["name"] for item in handler.responses[-1][1]["data"]["items"]], ["mine"])
        self.assertEqual(handler.responses[-1][1]["data"]["items"][0]["doc_ref"], ref)

    def test_documentation_routes_map_only_to_expected_live_docs_operations(self):
        ref = gw.felo_livedocs.make_ref("doc-123", "uid-a", self.secret)
        cases = [
            ("GET", "/resources?page=1&size=20", None, "/v2/livedocs/doc-123/resources?page=1&size=20"),
            ("GET", "/resources/res_1", None, "/v2/livedocs/doc-123/resources/res_1"),
            ("GET", "/resources/res_1/content", None, "/v2/livedocs/doc-123/resources/res_1/content"),
            ("GET", "/readme", None, "/v2/livedocs/doc-123/readme"),
            ("GET", "/tasks?status=0&page=1", None, "/v2/livedocs/doc-123/tasks?status=0&page=1"),
            ("GET", "/tasks/task_1/records", None, "/v2/livedocs/doc-123/tasks/task_1/records"),
            ("POST", "/resources/doc", {"content": "hello"}, "/v2/livedocs/doc-123/resources/doc"),
            ("POST", "/resources/urls", {"urls": [{"url": "https://example.com/page"}]}, "/v2/livedocs/doc-123/resources/urls"),
            ("POST", "/resources/retrieve", {"query": "find this"}, "/v2/livedocs/doc-123/resources/retrieve"),
            ("POST", "/resources/route", {"query": "find this"}, "/v2/livedocs/doc-123/resources/route"),
            ("POST", "/resources/ppt-retrieve", {"resource_id": "res_1", "page_number": 1, "query": "price"}, "/v2/livedocs/doc-123/resources/ppt-retrieve"),
            ("POST", "/readme/append", {"content": "section"}, "/v2/livedocs/doc-123/readme/append"),
            ("POST", "/tasks", {"title": "Task", "status": 0, "sort": 0}, "/v2/livedocs/doc-123/tasks"),
            ("POST", "/tasks/task_1/comments", {"content": "note"}, "/v2/livedocs/doc-123/tasks/task_1/comments"),
            ("PUT", "", {"name": "Updated"}, "/v2/livedocs/doc-123"),
            ("PUT", "/readme", {"content": "readme"}, "/v2/livedocs/doc-123/readme"),
            ("PUT", "/resources/res_1", {"title": "Updated"}, "/v2/livedocs/doc-123/resources/res_1"),
            ("PUT", "/resources/res_1/content", {"content": "new body"}, "/v2/livedocs/doc-123/resources/res_1/content"),
            ("PATCH", "/tasks/task_1", {"status": 1}, "/v2/livedocs/doc-123/tasks/task_1"),
            ("DELETE", "", None, "/v2/livedocs/doc-123"),
            ("DELETE", "/readme", None, "/v2/livedocs/doc-123/readme"),
            ("DELETE", "/resources/res_1", None, "/v2/livedocs/doc-123/resources/res_1"),
            ("DELETE", "/tasks/task_1", None, "/v2/livedocs/doc-123/tasks/task_1"),
        ]
        for method, suffix, body, expected_path in cases:
            with self.subTest(method=method, suffix=suffix):
                raw = b"" if body is None else json.dumps(body).encode()
                handler = FakeHandler(raw, doc_ref=ref)
                route = gw.felo_livedocs.BASE + suffix
                upstream = Mock(return_value=({"status": "ok", "data": {"ok": True}}, "application/json"))
                handled = gw.felo_livedocs.handle(handler, "uid-a", method, urlsplit(route), upstream, gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key")
                self.assertTrue(handled)
                self.assertEqual(handler.responses[-1][0], 200)
                upstream.assert_called_once_with(method, expected_path, body)

    def test_upload_doc_route_forwards_only_under_own_signed_ref(self):
        ref = gw.felo_livedocs.make_ref("doc-123", "uid-a", self.secret)
        boundary = "upload-boundary"
        body = b"--upload-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"notes.txt\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n--upload-boundary\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nNotes\r\n--upload-boundary--\r\n"
        handler = FakeHandler(body, f"multipart/form-data; boundary={boundary}", ref)
        with patch.object(gw.felo_livedocs, "_felo_upload", return_value={"status": "ok", "data": {"id": "res_1"}}) as upload:
            gw.felo_livedocs.handle(handler, "uid-a", "POST", urlsplit(gw.felo_livedocs.BASE + "/resources/upload-doc"), Mock(), gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key")
        self.assertEqual(handler.responses[-1][0], 200)
        self.assertEqual(upload.call_args.args[:5], ("/v2/livedocs/doc-123/resources/upload-doc", b"hello", "notes.txt", "text/plain", "Notes"))

    def test_invalid_task_status_is_rejected_before_upstream(self):
        ref = gw.felo_livedocs.make_ref("doc-123", "uid-a", self.secret)
        handler = FakeHandler(doc_ref=ref)
        upstream = Mock()
        gw.felo_livedocs.handle(handler, "uid-a", "GET", urlsplit(gw.felo_livedocs.BASE + "/tasks?status=9"), upstream, gw.FeloRequestError, gw._send_felo_error, self.secret, "mock-key")
        upstream.assert_not_called()
        self.assertEqual(handler.responses[-1], (400, {"error": "invalid_status"}))

    def test_upload_parser_accepts_one_bounded_file_and_title(self):
        boundary = "test-boundary"
        body = (
            b"--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"notes.txt\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n"
            b"--test-boundary\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nNotes\r\n--test-boundary--\r\n"
        )
        handler = FakeHandler(body, f"multipart/form-data; boundary={boundary}")
        parsed = gw.felo_livedocs._multipart_file(handler)
        self.assertEqual(parsed, (b"hello", "notes.txt", "text/plain", "Notes"))
        self.assertEqual(handler.responses, [])


if __name__ == "__main__":
    unittest.main()
