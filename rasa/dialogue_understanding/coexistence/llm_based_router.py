from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import structlog
from jinja2 import Template

import rasa.shared.utils.io
from rasa.dialogue_understanding.coexistence.constants import (
    CALM_ENTRY,
    NLU_ENTRY,
    STICKY,
    NON_STICKY,
)
from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.dialogue_understanding.generator.constants import LLM_CONFIG_KEY
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    ROUTE_TO_CALM_SLOT,
    PROMPT_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    OPENAI_PROVIDER,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException, FileIOException
from rasa.shared.nlu.constants import COMMANDS, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    get_prompt_template,
    llm_factory,
    try_instantiate_llm_client,
)
from rasa.utils.log_utils import log_llm

LLM_BASED_ROUTER_PROMPT_FILE_NAME = "llm_based_router_prompt.jinja2"
DEFAULT_COMMAND_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.coexistence", "router_template.jinja2"
)

# Token ids for gpt 3.5 and gpt 4 corresponding to space + capitalized Letter
A_TO_C_TOKEN_IDS_CHATGPT = [
    362,  # " A"
    426,  # " B"
    356,  # " C"
]

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_CHAT_MODEL_NAME,
    TIMEOUT_CONFIG_KEY: 7,
    "temperature": 0.0,
    "max_tokens": 1,
    "logit_bias": {str(token_id): 100 for token_id in A_TO_C_TOKEN_IDS_CHATGPT},
}

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COEXISTENCE_ROUTER,
    ],
    is_trainable=True,
)
class LLMBasedRouter(GraphComponent):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            PROMPT_CONFIG_KEY: None,
            CALM_ENTRY: {STICKY: None},
            NLU_ENTRY: {
                NON_STICKY: "handles chitchat",
                STICKY: "handles everything else",
            },
            LLM_CONFIG_KEY: None,
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.config = {**self.get_default_config(), **config}

        self.prompt_template = (
            prompt_template
            or get_prompt_template(
                config.get(PROMPT_CONFIG_KEY),
                DEFAULT_COMMAND_PROMPT_TEMPLATE,
            ).strip()
        )

        self._model_storage = model_storage
        self._resource = resource
        self.validate_config()

    def validate_config(self) -> None:
        """Validate the config of the router."""
        if (
            self.config[CALM_ENTRY] is None
            or not isinstance(self.config[CALM_ENTRY], dict)
            or STICKY not in self.config[CALM_ENTRY]
            or self.config[CALM_ENTRY][STICKY] is None
        ):
            raise ValueError(
                "The LLMBasedRouter component needs a proper "
                "description of the capabilities implemented in the CALM "
                "part of the bot."
            )

    def persist(self) -> None:
        """Persist this component to disk for future loading."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / LLM_BASED_ROUTER_PROMPT_FILE_NAME
            )

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        # Validate llm configuration
        try_instantiate_llm_client(
            self.config.get(LLM_CONFIG_KEY),
            DEFAULT_LLM_CONFIG,
            "llm_based_router.train",
            "LLMBasedRouter",
        )

        self.persist()
        return self._resource

    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMBasedRouter":
        """Loads trained component (see parent class for full docstring)."""
        prompt_template = None
        try:
            with model_storage.read_from(resource) as path:
                prompt_template = rasa.shared.utils.io.read_file(
                    path / LLM_BASED_ROUTER_PROMPT_FILE_NAME
                )
        except (FileNotFoundError, FileIOException) as e:
            structlogger.warning(
                "llm_based_router.load.failed", error=e, resource=resource.name
            )

        return cls(config, model_storage, resource, prompt_template=prompt_template)

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LLMBasedRouter:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    async def process(
        self,
        messages: List[Message],
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Message]:
        """Process a list of messages."""
        if tracker is None:
            # cannot do anything if there is no tracker (happens during 'rasa test nlu')
            return messages

        for message in messages:
            commands = await self.predict_commands(message, tracker)
            commands_dicts = [command.as_dict() for command in commands]
            message.set(COMMANDS, commands_dicts, add_to_output=True)

        return messages

    async def predict_commands(
        self,
        message: Message,
        tracker: DialogueStateTracker,
    ) -> List[Command]:
        if not tracker.has_coexistence_routing_slot:
            raise InvalidConfigException(
                f"Tried to run the LLMBasedRouter component "
                f"without the slot to track coexistence routing ({ROUTE_TO_CALM_SLOT})."
            )

        route_session_to_calm = tracker.get_slot(ROUTE_TO_CALM_SLOT)
        if route_session_to_calm is None:
            prompt = self.render_template(message)
            log_llm(
                logger=structlogger,
                log_module="LLMBasedRouter",
                log_event="llm_based_router.prompt_rendered",
                prompt=prompt,
            )
            # generating answer
            answer = await self._generate_answer_using_llm(prompt)
            log_llm(
                logger=structlogger,
                log_module="LLMBasedRouter",
                log_event="llm_based_router.llm_answer",
                answer=answer,
            )
            commands = self.parse_answer(answer)
            log_llm(
                logger=structlogger,
                log_module="LLMBasedRouter",
                log_event="llm_based_router.final_commands",
                commands=commands,
            )
            return commands
        elif route_session_to_calm is True:
            # don't set any commands so that a `LLMBasedCommandGenerator` is triggered
            # and can predict the actual commands.
            return []
        else:
            # If the session is assigned to DM1 add a `NoopCommand` to silence
            # the other command generators.
            return [NoopCommand()]

    @staticmethod
    def parse_answer(answer: Optional[str]) -> List[Command]:
        if answer is None:
            structlogger.warn(
                "llm_based_router.parse_answer.invalid_answer", answer=answer
            )
            return [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]

        # removing any whitespaces from the token
        answer = answer.strip()

        # to calm
        if answer == "A":
            return []
        # to dm1
        elif answer == "C":
            return [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]
        # to dm1 for a single chitchat turn
        elif answer == "B":
            return [NoopCommand()]
        else:
            structlogger.warn(
                "llm_based_router.parse_answer.invalid_answer", answer=answer
            )
            return [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]

    def render_template(self, message: Message) -> str:
        inputs = {
            "user_message": message.get(TEXT),
            f"{CALM_ENTRY}_{STICKY}": self.config[CALM_ENTRY][STICKY],
            f"{NLU_ENTRY}_{STICKY}": self.config[NLU_ENTRY][STICKY],
            f"{NLU_ENTRY}_{NON_STICKY}": self.config[NLU_ENTRY][NON_STICKY],
        }

        return Template(self.prompt_template).render(**inputs)

    async def _generate_answer_using_llm(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The generated text.
        """
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)

        try:
            llm_response = await llm.acompletion(prompt)
            return llm_response.choices[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("llm_based_router.llm.error", error=e)
            return None
