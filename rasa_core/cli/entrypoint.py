import logging

from rasa_core import run
from rasa_core import utils
from rasa_core.broker import PikaProducer
from rasa_core.interpreter import RasaNLUHttpInterpreter
from rasa_core.run import AvailableEndpoints
from rasa_core.tracker_store import MongoTrackerStore
from rasa_core.utils import EndpointConfig

logger = logging.getLogger()  # get the root logger

if __name__ == '__main__':
    # Setting up logfile
    logging.basicConfig(level=config.log_level)
    utils.configure_file_logging(logging.DEBUG, config.core_logfile)

    _nlg_endpoint = EndpointConfig(
            config.platform_host + "/nlg",
            token=config.platform_token)

    _model_endpoint = EndpointConfig(
            config.rasa_core_model_server,
            token=config.platform_token)

    _nlu_endpoint = EndpointConfig(
            config.nlu_host,
            token=config.nlu_token)

    _action_endpoint = EndpointConfig(
            config.user_app_host + "/webhook",
            token=config.user_app_token)

    _endpoints = AvailableEndpoints(nlg=_nlg_endpoint,
                                    nlu=None,
                                    action=_action_endpoint,
                                    model=_model_endpoint)

    _interpreter = RasaNLUHttpInterpreter(endpoint=_nlu_endpoint,
                                          project_name=config.project_name)

    _event_broker = PikaProducer(config.rabbitmq_host,
                                 config.rabbitmq_username,
                                 config.rabbitmq_password)

    tracker_store = MongoTrackerStore(domain=None,
                                      host=config.mongo_host,
                                      db=config.mongo_database,
                                      username=config.mongo_username,
                                      password=config.mongo_password,
                                      collection="conversations",
                                      event_broker=_event_broker)

    _agent = run.load_agent(
            config.core_model_dir,
            interpreter=_interpreter,
            endpoints=_endpoints,
            tracker_store=tracker_store,
            wait_time_between_pulls=config.core_model_pull_interval)

    # Start handling the input channel
    run.serve_application(_agent,
                          credentials_file=config.channel_credentials_file,
                          port=config.self_port,
                          auth_token=config.core_token,
                          jwt_secret=config.jwt_secret,
                          jwt_method='HS256',
                          cors="*")
