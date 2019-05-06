import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":  # pragma: no cover
    raise RuntimeError(
        "Calling `rasa.core.evaluate` is deprecated. "
        "Please use `rasa.core.test` instead."
    )
