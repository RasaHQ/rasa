import logging
import typing

import rasa.shared.utils.common

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    pass

# backwards compatibility
run = rasa.run
