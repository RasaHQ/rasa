from typing import Protocol, Dict, List, runtime_checkable, Union

from rasa.shared.providers.llm.llm_response import LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for an LLM client that specifies the interface for interacting
    with the API.
    """

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        """
        Initializes the llm client with the given configuration.

        This class method should be implemented to parse the given
        configuration and create an instance of an llm client.
        """
        ...

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for that the llm client is initialized with.

        This property should be implemented to return a dictionary containing
        the configuration settings for the llm client.
        """
        ...

    def completion(self, messages: Union[List[str], str]) -> LLMResponse:
        """
        Synchronously generate completions for given list of messages.

        This method should be implemented to take a list of messages (as
        strings) and return a list of completions (as strings).

        Args:
            messages: List of messages or a single message to generate the
                completion for.
        Returns:
            LLMResponse
        """
        ...

    async def acompletion(self, messages: Union[List[str], str]) -> LLMResponse:
        """
        Asynchronously generate completions for given list of messages.

        This method should be implemented to take a list of messages (as
        strings) and return a list of completions (as strings).

        Args:
            messages: List of messages or a single message to generate the
                completion for.
        Returns:
            LLMResponse
        """
        ...

    def validate_client_setup(self, *args, **kwargs) -> None:  # type: ignore
        """
        Perform client setup validation.

        This method should be implemented to validate whether the client can be
        used with the parameters provided through configuration or environment
        variables.

        If there are any issues, the client should raise
        ProviderClientValidationError.

        If no validation is needed, this check can simply pass.
        """
        ...
