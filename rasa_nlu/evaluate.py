import logging

import rasa_nlu.test as test

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # pragma: no cover
    logger.warning("Calling `rasa_nlu.evaluate` is deprecated. "
                   "Please use `rasa_nlu.test` instead.")
    test.main()
