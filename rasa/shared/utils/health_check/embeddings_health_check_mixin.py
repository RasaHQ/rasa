from typing import Any, Dict, Optional


class EmbeddingsHealthCheckMixin:
    """Mixin class that provides methods for performing embeddings health checks during
    training and inference within components.

    This mixin offers static methods that wrap the following health check functions:
    - `perform_embeddings_health_check`
    """

    @staticmethod
    def perform_embeddings_health_check(
        custom_embeddings_config: Optional[Dict[str, Any]],
        default_embeddings_config: Dict[str, Any],
        log_source_method: str,
        log_source_component: str,
    ) -> None:
        """Wraps the `perform_embeddings_health_check` function to enable
        tracing and instrumentation.
        """
        from rasa.shared.utils.health_check.health_check import (
            perform_embeddings_health_check,
        )

        perform_embeddings_health_check(
            custom_embeddings_config,
            default_embeddings_config,
            log_source_method,
            log_source_component,
        )
