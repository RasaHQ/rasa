from yarl import URL
import textwrap
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader


def latest_request(mocked, request_type, path):
    return mocked.requests.get((request_type, URL(path)))


def json_of_latest_request(r):
    return r[-1].kwargs["json"]


def flows_from_str(yaml_str: str) -> FlowsList:
    """Reads flows from a YAML string."""
    return YAMLFlowsReader.read_from_string(textwrap.dedent(yaml_str))
