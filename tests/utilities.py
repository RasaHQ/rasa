from typing import List, Text, Optional, Dict

from yarl import URL

from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.importers.importer import FlowSyncImporter


def latest_request(mocked, request_type, path):
    return mocked.requests.get((request_type, URL(path)))


def json_of_latest_request(r):
    return r[-1].kwargs["json"]


def flows_from_str_with_defaults(yaml_str: str) -> FlowsList:
    """Reads flows from a YAML string and includes buildin flows."""
    return FlowSyncImporter.merge_with_default_flows(flows_from_str(yaml_str))


def flows_default_domain() -> Domain:
    """Returns the default domain for the default flows."""
    return FlowSyncImporter.load_default_pattern_flows_domain()


def filter_logs(
    caplog: List[Dict],
    event: Optional[Text] = None,
    log_level: Optional[Text] = None,
    log_message_parts: Optional[List[Text]] = None,
) -> List[Dict]:
    return [
        log
        for log in caplog
        if (event is None or log["event"] == event)
        and (log_level is None or log["log_level"] == log_level)
        and (
            log_message_parts is None
            or all(part in log["event_info"] for part in log_message_parts)
        )
    ]
