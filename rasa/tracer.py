import logging
from jaeger_client import Config

class Tracer(object):
    def __init__(self, service_name="rasa"):
        logger = logging.getLogger(__name__)
        logging.getLogger("").handlers = []
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        config = Config(
            config={
                "sampler": {"type": "const", "param": 1},
                "logging": True,
            },
            service_name=service_name,
        )
        self.t = config.initialize_tracer()
