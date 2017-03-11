import os

import logging

from rasa_nlu.server import create_app

from rasa_nlu.config import RasaNLUConfig

if __name__ == '__main__':
    # Running in WSGI container, configuration will be loaded from the default location
    # There is no common support for WSGI runners to pass arguments to the application, hence we need to fallback to
    # a default location for the configuration where all the settings should be placed in.
    rasa_config = RasaNLUConfig(env_vars=os.environ)
    app = create_app(rasa_config)
    logging.info("Finished setting up application")
    app.run()
