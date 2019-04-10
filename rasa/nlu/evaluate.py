import logging


logger = logging.getLogger(__name__)


if __name__ == "__main__":  # pragma: no cover
    raise RuntimeError(
        "Calling `rasa.nlu.evaluate` directly is "
        "no longer supported. "
        "Please use `rasa test nlu` instead."
    )
