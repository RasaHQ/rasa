"""Mock-only tests for Felo gateway tool routing; never contacts Felo or spends credits."""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("RASA_AUTH_TOKEN", "test-only-not-a-real-secret")
try:
    import jwt  # noqa: F401
except ImportError:
    jwt_stub = types.ModuleType("jwt")
    jwt_stub.encode = lambda *args, **kwargs: "test-token"
    sys.modules["jwt"] = jwt_stub

MODULE_PATH = Path(__file__).with_name("secure_gateway.py")
spec = importlib.util.spec_from_file_location("secure_gateway_under_test", MODULE_PATH)
gw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gw
spec.loader.exec_module(gw)


class FeloToolTests(unittest.TestCase):
    def setUp(self):
        gw.FELO_TASK_OWNERS.clear()
        gw.FELO_THREAD_OWNERS.clear()
        gw.FELO_STREAM_OWNERS.clear()
        gw._FELO_X_REQUEST_TIMES.clear()

    def test_ordinary_text_stays_with_rasa_and_clear_natural_trigger_matches(self):
        with patch.object(gw, "_felo_request", return_value=({"status": "ok", "data": {"task_id": "task-a", "live_doc_short_id": "must-not-leak"}}, "application/json")) as upstream:
            self.assertIsNone(gw._felo_chat_tool("Hola Sara, ¿cómo estás?", "uid-a"))
            reply = gw._felo_chat_tool("Crea una presentación sobre energía solar", "uid-a")
        upstream.assert_called_once_with("POST", "/v2/ppts", {"query": "energía solar"})
        self.assertIn("vista previa", reply.lower())
        self.assertTrue(gw._owns(gw.FELO_TASK_OWNERS, "task-a", "uid-a"))
        self.assertFalse(gw._owns(gw.FELO_TASK_OWNERS, "task-a", "uid-b"))

    def test_llm_non_streaming_strips_tool_calls(self):
        def fake(method, path, body=None, **kwargs):
            self.assertEqual(path, "/api/v1/chat/completions")
            self.assertFalse(body["stream"])
            self.assertNotIn("tools", body)
            self.assertNotIn("tool_choice", body)
            self.assertEqual(body["model"], "model-test")
            return {"status": "ok", "choices": [{"message": {"content": "hello", "tool_calls": [{"id": "ignored"}]}}]}, "application/json"
        with patch.object(gw, "_felo_request", side_effect=fake) as upstream:
            result = gw._felo_llm("chat/completions", {
                "model": "model-test", "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function"}], "tool_choice": "auto", "stream": True,
            })
        self.assertEqual(gw._llm_text(result), "hello")
        upstream.assert_called_once()

    def test_x_results_clamped_to_five_and_uses_mock(self):
        records = [{"id": str(i)} for i in range(8)]
        with patch.object(gw, "_felo_request", return_value=({"status": "ok", "data": {"items": records}}, "application/json")) as upstream:
            result = gw._felo_x_request("tweet-search", {"query": "test", "limit": 500})
        self.assertEqual(upstream.call_args.args[2]["limit"], 5)
        self.assertEqual(len(result["data"]["items"]), 5)

    def test_cross_user_thread_and_unknown_livedoc_fail_closed(self):
        gw._remember_owner(gw.FELO_THREAD_OWNERS, "thread-a", "uid-a")
        self.assertTrue(gw._owns(gw.FELO_THREAD_OWNERS, "thread-a", "uid-a"))
        self.assertFalse(gw._owns(gw.FELO_THREAD_OWNERS, "thread-a", "uid-b"))
        self.assertFalse(gw._owns(gw.FELO_THREAD_OWNERS, "not-in-local-owner-store", "uid-a"))

    def test_openai_compatible_response_does_not_require_harness_envelope(self):
        class Response:
            headers = {"Content-Type": "application/json"}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit): return b'{"object":"list","data":[{"id":"model-a"}]}'
        with patch.object(gw, "FELO_API_KEY", "mock-key"), patch.object(gw.urllib.request, "urlopen", return_value=Response()):
            payload, _ = gw._felo_request("GET", "/api/v1/models")
        self.assertEqual(payload["data"][0]["id"], "model-a")

    def test_superagent_sse_is_parsed_without_leaking_stream_or_provider_errors(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __iter__(self):
                return iter([b"event: message\n", b'data: {"content":"safe answer"}\n', b"event: done\n", b"data: {}\n"])
        with patch.object(gw, "FELO_API_KEY", "mock-key"), patch.object(gw.urllib.request, "urlopen", return_value=Response()):
            self.assertEqual(gw._sse_text("owned-stream"), "safe answer")


if __name__ == "__main__":
    unittest.main()
