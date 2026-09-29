import io
import json
import unittest
from urllib.parse import parse_qs, urlsplit

import sara_memory


class SaraMemoryTests(unittest.TestCase):
    def test_personal_recall_recognizes_accents_and_name_questions(self):
        self.assertTrue(sara_memory.is_personal_recall_query("¿Recuerdas cómo me llamo?"))
        self.assertTrue(sara_memory.is_personal_recall_query("¿Qué sabes sobre mí?"))
        self.assertTrue(sara_memory.is_personal_recall_query("¿Cuál es mi nombre?"))
        self.assertFalse(sara_memory.is_personal_recall_query("¿Sabes cómo reparar Firebase para mí?"))

    def test_identity_questions_distinguish_sara_from_user(self):
        self.assertTrue(sara_memory.is_assistant_name_query("¿Cuál es tu nombre?"))
        self.assertTrue(sara_memory.is_assistant_name_query("¿Cómo te llamas?"))
        self.assertTrue(sara_memory.is_assistant_name_query("¿Tu nombre es Sara?"))
        self.assertFalse(sara_memory.is_user_name_query("¿Cuál es tu nombre?"))
        self.assertFalse(sara_memory.is_personal_recall_query("¿Cuál es tu nombre?"))
        self.assertTrue(sara_memory.is_user_name_query("¿Cómo me llamo?"))
        self.assertTrue(sara_memory.is_user_name_query("¿Cuál es mi nombre?"))
        self.assertFalse(sara_memory.is_assistant_name_query("¿Cómo me llamo?"))

    def test_assistant_name_reply_is_fixed(self):
        self.assertEqual(
            sara_memory.assistant_name_reply(),
            "Me llamo Sara, soy la asistente de OmniStudio.",
        )

    def test_user_name_reply_quotes_only_the_top_relevant_note(self):
        reply = sara_memory.format_user_name_reply("• Me llamo Cristopher\n• Mi correo es cris@example.com")
        self.assertEqual(reply, "Según la nota que guardaste: Me llamo Cristopher.")
        self.assertNotIn("Sara", reply)

    def test_user_name_reply_does_not_guess_when_no_note_exists(self):
        reply = sara_memory.format_user_name_reply("")
        self.assertIn("no voy a adivinarlo", reply)
        self.assertNotIn("Sara", reply)

    def test_name_question_selects_name_note_without_unrelated_email(self):
        entries = [
            {"ownerUid": "u1", "text": "Mi correo electrónico es cris@example.com", "createdAt": 20},
            {"ownerUid": "u1", "text": "Me llamo Cristopher", "createdAt": 10},
        ]
        result = sara_memory.select_relevant_context(entries, "¿Recuerdas cómo me llamo?")
        self.assertIn("Me llamo Cristopher", result)
        self.assertNotIn("correo", result)

    def test_name_query_matches_third_person_se_llama_note(self):
        entries = [
            {"ownerUid": "u1", "text": "Mi correo electrónico es cris@example.com", "createdAt": 20},
            {"ownerUid": "u1", "text": "El usuario se llama Cristopher Cook Gonzalez", "createdAt": 10},
        ]
        for query in ("¿Cómo me llamo?", "¿Cuál es mi nombre?", "¿Cuál es mi nombre completo?"):
            with self.subTest(query=query):
                result = sara_memory.select_relevant_context(entries, query)
                self.assertIn("El usuario se llama Cristopher Cook Gonzalez", result)
                self.assertNotIn("correo", result)

    def test_personal_recall_falls_back_to_recent_notes_when_wording_does_not_match(self):
        entries = [
            {"ownerUid": "u1", "text": "Preferred language: Spanish", "createdAt": 20},
            {"ownerUid": "u1", "text": "Cristopher Cook Gonzalez", "createdAt": 10},
        ]
        result = sara_memory.select_relevant_context(entries, "¿Cómo me llamo?")
        self.assertIn("Cristopher Cook Gonzalez", result)

    def test_overview_selects_recent_approved_notes(self):
        entries = [
            {"ownerUid": "u1", "text": "Dato anterior", "createdAt": 1},
            {"ownerUid": "u1", "text": "Me llamo Cristopher", "createdAt": 3},
            {"ownerUid": "u1", "text": "Uso OmniStudio", "createdAt": 2},
        ]
        result = sara_memory.select_relevant_context(entries, "¿Qué sabes sobre mí?")
        self.assertIn("Me llamo Cristopher", result)
        self.assertIn("Uso OmniStudio", result)
        self.assertIn("Dato anterior", result)

    def test_rejects_non_firebase_host_before_sending_token(self):
        with self.assertRaises(sara_memory.SaraMemoryFetchError) as error:
            sara_memory.fetch_relevant_context(
                "https://attacker.example", "uid-123", "fake-token", "¿Qué sabes sobre mí?"
            )
        self.assertEqual(error.exception.code, "invalid_database_url")

    def test_fetch_uses_uid_and_user_id_token_and_filters_owner(self):
        uid, token = "uid-123", "fake-id-token-for-test"
        payload = {
            "a": {"ownerUid": uid, "text": "Me llamo Cristopher", "createdAt": 1},
            "b": {"ownerUid": "other-user", "text": "Otro usuario", "createdAt": 2},
        }
        calls = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return json.dumps(payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return Response()

        stats = []
        result = sara_memory.fetch_relevant_context(
            "https://example.firebaseio.com", uid, token, "¿Cómo me llamo?",
            urlopen=fake_urlopen, status_callback=stats.append
        )
        request, timeout = calls[0]
        parsed = urlsplit(request.full_url)
        self.assertEqual(parsed.path, "/sara_knowledge/uid-123.json")
        self.assertEqual(parse_qs(parsed.query)["auth"], [token])
        self.assertEqual(timeout, 8)
        self.assertIn("Me llamo Cristopher", result)
        self.assertNotIn("Otro usuario", result)
        self.assertEqual(stats, [{
            "records": 2, "dict_records": 2, "owner_match": 1, "owner_missing": 0,
            "owner_mismatch": 1, "nonempty_text": 1, "selected": 1,
        }])
        self.assertNotIn("text", stats[0])

    def test_fetch_accepts_legacy_note_without_owner_uid(self):
        uid, token = "uid-123", "fake-id-token-for-test"
        payload = {
            "legacy": {"text": "Cristopher Cook Gonzalez", "createdAt": 1},
            "foreign": {"ownerUid": "other-user", "text": "Otro usuario", "createdAt": 2},
        }

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return json.dumps(payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            return Response()

        result = sara_memory.fetch_relevant_context(
            "https://example.firebaseio.com", uid, token, "¿Cómo me llamo?", urlopen=fake_urlopen
        )
        self.assertIn("Cristopher Cook Gonzalez", result)
        self.assertNotIn("Otro usuario", result)


if __name__ == "__main__":
    unittest.main()

