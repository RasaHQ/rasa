import logging

import rasa_core.test as test

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # pragma: no cover
    logger.warning("Calling `rasa_core.evaluate` is deprecated. "
                   "Please use `rasa_core.test` instead.")
    test.main()
