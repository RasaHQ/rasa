from typing import Protocol, Dict, List

from rasa.shared.providers.llm.llm_response import LLMResponse


class LLMClient(Protocol):
    """
    Protocol for an LLM client that specifies the interface for interacting
    with the API.
    """

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for that the llm client is initialized with.

        This property should be implemented to return a dictionary containing
        the configuration settings for the llm client.
        """
        ...

    @property
    def model(self) -> str:
        """Returns the model name.

        This property should be implemented to return a string representing the
        model name client sends the API requests to.
        """
        ...

    @property
    def provider(self) -> str:
        """Returns the provider name.

        This property should be implemented to return a string representing the
        provider name.
        """
        ...

    def completion(self, messages: List[str]) -> LLMResponse:
        """
        Synchronously generate completions for given list of messages.

        This method should be implemented to take a list of messages (as
        strings) and return a list of completions (as strings).

        Args:
            messages: List of messages to generate the completions for.
        Returns:
            List of message completions.
        """
        ...

    async def acompletion(self, messages: List[str]) -> LLMResponse:
        """
        Asynchronously generate completions for given list of messages.

        This method should be implemented to take a list of messages (as
        strings) and return a list of completions (as strings).

        Args:
            messages: List of messages to generate the completion for.
        Returns:
            List of message completions.
        """
        ...
