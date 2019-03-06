import logging

from rasa_nlu.test import main

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # pragma: no cover
    logger.warning("Calling `rasa_nlu.evaluate` is deprecated. "
                   "Please use `rasa_nlu.test` instead.")
    main()
