import dataclasses
import io
from typing import Dict, List, Optional, Text

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
    log_contains_all_message_parts: bool = True,
) -> List[Dict]:
    """Filters structlog logs based on specified criteria:

    Args:
        caplog:
            List of structlog logs
        event:
            The specific event type in log.
        log_level:
            Log level
        log_message_parts:
            Parts of the user-facing message, found within the `event_info` key of a
            log.
        log_contains_all_message_parts:
            Flag determining the filtering logic for message parts. If True, a log
            entry must contain all specified parts to be included. If False, the
            presence of any specified part suffices for inclusion.

    Returns:
        A list of logs that matches the filtering criteria.
    """

    def contains_message_parts(log: Dict) -> bool:
        contains_parts = [
            part in log.get("event_info", "") for part in log_message_parts
        ]
        if log_contains_all_message_parts:
            return all(contains_parts)
        return any(contains_parts)

    filtered_logs = []

    for log in caplog:
        matches_event = event is None or log["event"] == event
        matches_log_level = log_level is None or log["log_level"] == log_level
        matches_message_parts = log_message_parts is None or contains_message_parts(log)

        if matches_event and matches_log_level and matches_message_parts:
            filtered_logs.append(log)

    return filtered_logs


@dataclasses.dataclass
class TarFileEntry:
    """This class is used to represent a file entry in a tar file."""

    name: str
    data: bytes


def create_tar_archive_in_bytes(input_file_entries: List[TarFileEntry]) -> bytes:
    """Creates a tar archive in bytes format.

    Args:
        input_file_entries: List of TarFileEntry objects representing the files to be
        included in the tar archive.

    Returns:
        Bytes format of the tar archive
    """
    byte_array = bytes()
    file_like_object = io.BytesIO(byte_array)

    import tarfile

    with tarfile.open(fileobj=file_like_object, mode="w:gz") as tar:
        for item in input_file_entries:
            info = tarfile.TarInfo(name=item.name)
            info.size = len(item.data)
            tar.addfile(info, io.BytesIO(item.data))

    return file_like_object.getvalue()
