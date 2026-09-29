import json
import unittest
from unittest.mock import patch

import groq_service


class GroqWorkspaceConnectorTests(unittest.TestCase):
    def test_only_selected_read_only_connectors_are_sent(self):
        with patch.object(groq_service, "_request", return_value={"output_text": "Encontré dos correos."}) as request:
            answer = groq_service.workspace_connector_reply(
                "Busca mis correos recientes",
                {"gmail": "fake-google-access-token"},
            )
        self.assertEqual(answer, "Encontré dos correos.")
        self.assertEqual(request.call_args.args[0], groq_service.RESPONSES_URL)
        payload = json.loads(request.call_args.args[1])
        self.assertEqual(payload["model"], groq_service.TOOL_MODEL)
        self.assertEqual(payload["tools"], [{
            "type": "mcp",
            "server_label": "Gmail",
            "connector_id": "connector_gmail",
            "authorization": "fake-google-access-token",
            "require_approval": "never",
        }])
        self.assertIn("no confiable", payload["instructions"])

    def test_all_three_supported_connectors_use_groq_ids(self):
        with patch.object(groq_service, "_request", return_value={"output_text": "Listo."}) as request:
            groq_service.workspace_connector_reply(
                "Revisa los elementos recientes",
                {"gmail": "g1", "calendar": "g2", "drive": "g3"},
            )
        payload = json.loads(request.call_args.args[1])
        self.assertEqual(
            {tool["connector_id"] for tool in payload["tools"]},
            {"connector_gmail", "connector_googlecalendar", "connector_googledrive"},
        )

    def test_unknown_connector_is_rejected_before_provider_call(self):
        with patch.object(groq_service, "_request") as request:
            with self.assertRaises(groq_service.GroqServiceError) as error:
                groq_service.workspace_connector_reply("Busca algo", {"calendar_write": "fake-token"})
        self.assertEqual(error.exception.code, "invalid_workspace_connectors")
        request.assert_not_called()

    def test_access_token_is_required_and_never_in_error_text(self):
        with self.assertRaises(groq_service.GroqServiceError) as error:
            groq_service.workspace_connector_reply("Busca algo", {"gmail": "  "})
        self.assertEqual(error.exception.code, "invalid_workspace_access_token")
        self.assertNotIn("fake", str(error.exception))

    def test_long_prompt_is_rejected(self):
        with patch.object(groq_service, "_request") as request:
            with self.assertRaises(groq_service.GroqServiceError) as error:
                groq_service.workspace_connector_reply("x" * 4001, {"gmail": "fake-token"})
        self.assertEqual(error.exception.code, "valid_workspace_query_required")
        request.assert_not_called()

    def test_output_text_is_extracted_from_responses_shape(self):
        with patch.object(groq_service, "_request", return_value={
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Evento encontrado."}],
            }]
        }):
            answer = groq_service.workspace_connector_reply("Busca mi calendario", {"calendar": "fake-token"})
        self.assertEqual(answer, "Evento encontrado.")


if __name__ == "__main__":
    unittest.main()
