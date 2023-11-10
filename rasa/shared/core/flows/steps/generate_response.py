from __future__ import annotations

from dataclasses import dataclass
from typing import Text, Optional, Dict, Any
import structlog
from jinja2 import Template
from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import (
    llm_factory,
    tracker_as_readable_transcript,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
)

from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_TEMPERATURE,
    DEFAULT_OPENAI_GENERATE_MODEL_NAME,
)

DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 5,
    "temperature": DEFAULT_OPENAI_TEMPERATURE,
    "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
}
structlogger = structlog.get_logger()


@dataclass
class GenerateResponseFlowStep(FlowStep):
    """A flow step that creates a free-form bot utterance using an LLM."""

    generation_prompt: Text
    """The prompt template of the flow step."""
    llm_config: Optional[Dict[Text, Any]] = None
    """The LLM configuration of the flow step."""

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> GenerateResponseFlowStep:
        """Create a GenerateResponseFlowStep from serialized data

        Args:
            data: data for a GenerateResponseFlowStep in a serialized format

        Returns:
            A GenerateResponseFlowStep object
        """
        base = super().from_json(data)
        return GenerateResponseFlowStep(
            generation_prompt=data["generation_prompt"],
            llm_config=data.get("llm"),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the GenerateResponseFlowStep object.

        Returns:
            the GenerateResponseFlowStep object as serialized data.
        """
        data = super().as_json()
        data["generation_prompt"] = self.generation_prompt
        if self.llm_config:
            data["llm"] = self.llm_config

        return data

    @staticmethod
    def create_prompt_template_inputs(tracker: DialogueStateTracker) -> Dict[str, Any]:
        """Generate the prompt template inputs."""
        latest_message = tracker.latest_message.text if tracker.latest_message else ""
        inputs = {
            "history": tracker_as_readable_transcript(tracker, max_turns=5),
            "latest_user_message": latest_message,
        }
        inputs.update(tracker.current_slot_values())
        return inputs

    def generate(self, tracker: DialogueStateTracker) -> Optional[str]:
        """Generates a response for the given tracker.

        Args:
            tracker: The tracker to generate a response for.

        Returns:
            The generated response.
        """
        template_inputs = self.create_prompt_template_inputs(tracker)
        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)
        prompt = Template(self.generation_prompt).render(template_inputs)

        try:
            return llm(prompt)
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error(
                "flow.generate_step.llm.error", error=e, step=self.id, prompt=prompt
            )
            return None

    @property
    def default_id_postfix(self) -> str:
        return "generate"
