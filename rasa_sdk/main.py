import logging

from gevent.pywsgi import WSGIServer

from rasa_core import utils
from rasa_sdk.endpoint import endpoint_app
from rasa_sdk.executor import CoreEndpointConfig

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    utils.configure_colored_logging("DEBUG")

    core_endpoint = CoreEndpointConfig("localhost:5005")
    app = endpoint_app(action_package_name="myactions",
                       core_endpoint=core_endpoint)

    http_server = WSGIServer(('0.0.0.0', 5055), app)

    http_server.start()
    logger.info("Action endpoint is up and running.")

    http_server.serve_forever()
