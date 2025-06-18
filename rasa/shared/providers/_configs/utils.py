import structlog

structlogger = structlog.get_logger()


def get_provider_prefixed_model_name(provider: str, model: str) -> str:
    """
    Returns the model name with the provider prefixed.

    Args:
        provider: The provider of the model.
        model: The model name.

    Returns:
        The model name with the provider prefixed.
    """
    if model and f"{provider}/" not in model:
        return f"{provider}/{model}"
    return model
