"""
LLM Providers - OpenAI, Claude, and base provider
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self._client = None

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        super().__init__(api_key, model)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from env if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            logger.info(f"✅ OpenAI client initialized with model: {self.model}")
        except ImportError:
            logger.error("❌ OpenAI package not installed. Run: pip install openai")
            self._client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            self._client = None

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from OpenAI

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text response
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client not available")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )

            generated_text = response.choices[0].message.content

            # Log token usage
            usage = response.usage
            logger.info(
                f"🤖 OpenAI response generated | "
                f"Tokens: {usage.total_tokens} "
                f"(prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens})"
            )

            return generated_text

        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self._client is not None and self.api_key is not None


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        super().__init__(api_key, model)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from env if not provided
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Claude client"""
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
            logger.info(f"✅ Claude client initialized with model: {self.model}")
        except ImportError:
            logger.error("❌ Anthropic package not installed. Run: pip install anthropic")
            self._client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Claude: {e}")
            self._client = None

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from Claude

        Args:
            messages: List of message dicts with 'role' and 'content'
                     First message should be 'system' role (will be extracted)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text response
        """
        if not self.is_available():
            raise RuntimeError("Claude client not available")

        try:
            # Claude API separates system message from conversation
            system_message = ""
            conversation_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append(msg)

            # Ensure alternating user/assistant messages
            # If first message is not user, prepend a user message
            if conversation_messages and conversation_messages[0]["role"] != "user":
                conversation_messages.insert(0, {
                    "role": "user",
                    "content": "Hello"
                })

            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                system=system_message if system_message else None,
                messages=conversation_messages,
                **kwargs
            )

            generated_text = response.content[0].text

            # Log token usage
            usage = response.usage
            logger.info(
                f"🤖 Claude response generated | "
                f"Tokens: input={usage.input_tokens}, "
                f"output={usage.output_tokens}"
            )

            return generated_text

        except Exception as e:
            logger.error(f"❌ Claude generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Claude is available"""
        return self._client is not None and self.api_key is not None


def get_provider(provider_name: str = "openai", **kwargs) -> LLMProvider:
    """
    Factory function to get LLM provider

    Args:
        provider_name: 'openai' or 'claude'
        **kwargs: Provider-specific arguments

    Returns:
        LLMProvider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(providers.keys())}"
        )

    return provider_class(**kwargs)
