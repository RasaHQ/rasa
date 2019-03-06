import logging

from rasa_core.test import main

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # pragma: no cover
    logger.warning("Calling `rasa_core.evaluate` is deprecated. "
                   "Please use `rasa_core.test` instead.")
    main()
