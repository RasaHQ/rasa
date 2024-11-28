from typing import Optional, Dict, Any


class EmbeddingsHealthCheckMixin:
    """Mixin class that provides methods for performing embeddings health checks during
    training and inference within components.

    This mixin offers static methods that wrap the following health check functions:
    - `perform_training_time_embeddings_health_check`, and
    - `perform_inference_time_embeddings_health_check`
    """

    @staticmethod
    def perform_training_time_embeddings_health_check(
        custom_embeddings_config: Optional[Dict[str, Any]],
        default_embeddings_config: Dict[str, Any],
        log_source_method: str,
        log_source_component: str,
    ) -> Optional[str]:
        """Wraps the `perform_training_time_embeddings_health_check` function to enable
        tracing and instrumentation."""
        from rasa.shared.utils.health_check.health_check import (
            perform_training_time_embeddings_health_check,
        )

        return perform_training_time_embeddings_health_check(
            custom_embeddings_config,
            default_embeddings_config,
            log_source_method,
            log_source_component,
        )

    @staticmethod
    def perform_inference_time_embeddings_health_check(
        custom_embeddings_config: Optional[Dict[str, Any]],
        default_embeddings_config: Dict[str, Any],
        train_model_name: str,
        log_source_method: str,
        log_source_component: str,
    ) -> None:
        """Wraps the `perform_inference_time_embeddings_health_check` function to enable
        tracing and instrumentation."""
        from rasa.shared.utils.health_check.health_check import (
            perform_inference_time_embeddings_health_check,
        )

        perform_inference_time_embeddings_health_check(
            custom_embeddings_config,
            default_embeddings_config,
            train_model_name,
            log_source_method,
            log_source_component,
        )
