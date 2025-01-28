from dataclasses import fields

from rasa.shared.providers.llm.llm_response import LLMResponse


class TestLLMResponse:
    def test_ensure_llm_response_with_llm_response(
        self, llm_response_object: LLMResponse
    ):
        result = LLMResponse.ensure_llm_response(llm_response_object)
        assert result == llm_response_object

    def test_ensure_llm_response_with_string(self):
        response = "test_response"
        result = LLMResponse.ensure_llm_response(response)
        class_fields = fields(LLMResponse)
        empty_names = [field.name for field in class_fields if field.name != "choices"]
        for field in empty_names:
            assert getattr(result, field) is None
        assert result.choices == [response]
