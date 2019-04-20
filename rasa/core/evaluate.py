import logging

from rasa.core.test import main

logger = logging.getLogger(__name__)

if __name__ == "__main__":  # pragma: no cover
    logger.warning(
        "Calling `rasa.core.evaluate` is deprecated. "
        "Please use `rasa.core.test` instead."
    )
    main()
