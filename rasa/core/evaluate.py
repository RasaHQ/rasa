import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":  # pragma: no cover
    raise RuntimeError(
        "Calling `rasa.core.evaluate` directly is no longer supported. Please use "
        "`rasa test` to test a combined Core and NLU model or `rasa test core` "
        "to test a Core model."
    )
