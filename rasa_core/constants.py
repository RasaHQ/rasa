from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_URL = "http://localhost:{}".format(DEFAULT_SERVER_PORT)

MINIMUM_COMPATIBLE_VERSION = "0.11.0"

DOCS_BASE_URL = "https://rasa.com/docs/core"

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.0

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.0

DEFAULT_FALLBACK_ACTION = "action_default_fallback"

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes
