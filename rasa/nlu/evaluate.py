import logging

from rasa.nlu.test import main

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # pragma: no cover
    logger.warning("Calling `rasa.nlu.evaluate` is deprecated. "
                   "Please use `rasa.nlu.test` instead.")
    main()
