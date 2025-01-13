import logging
from pathlib import Path
from typing import Any, Dict, List, Text

import requests

from rasa.markers.validate import validate_marker_file
from rasa.shared.core.domain import Domain
from rasa.shared.utils.cli import print_error_and_exit
from rasa.utils.yaml import (
    YAML_CONFIGS,
    collect_configs_from_yaml_files,
    collect_yaml_files_from_path,
)

PATTERNS_PATH = "/api/v1/patterns"

log = logging.getLogger(__name__)


def upload(url: Text, domain: Domain, markers_path: Path) -> None:
    validate_marker_file(domain, markers_path)
    yaml_files = collect_yaml_files_from_path(markers_path)
    yaml_configs = collect_configs_from_yaml_files(yaml_files)

    marker_json_list = _convert_yaml_to_json(yaml_configs)

    patterns_url = f"{url}{PATTERNS_PATH}"
    try:
        response = requests.post(f"{patterns_url}", json={"patterns": marker_json_list})
    except requests.exceptions.ConnectionError:
        print_error_and_exit(
            f"Failed to connect to Rasa Pro Services at {patterns_url}. "
            f"Make sure the server is running and the correct URL is configured."
        )
        # this shouldn't be required but I need to add this for the test
        # test_upload::test_upload_with_connection_error to pass
        return

    if response.status_code == 200:
        patterns = response.json()
        log.info(
            f"Successfully uploaded markers to {url}"
            f"Total count: {patterns.get('count')}\n"
            f"Updated: {patterns.get('updated')}\n"
            f"Inserted: {patterns.get('inserted')}\n"
            f"Deleted: {patterns.get('deleted')}\n"
        )
    else:
        print_error_and_exit(
            f"Failed to upload markers to {patterns_url}. "
            f"Status Code: {response.status_code} "
            f"Response: {response.text}"
        )


def _convert_marker_config_to_json(
    marker_pattern_name: Text, config: Dict[Text, Any]
) -> Dict[Text, Any]:
    description = config.pop("description", None)
    return {
        "name": marker_pattern_name,
        "description": description,
        "config": config,
    }


def _convert_yaml_to_json(yaml_configs: YAML_CONFIGS) -> List[Dict]:
    marker_json_list: List[Dict] = []
    for _, yaml_config in yaml_configs.items():
        for marker_pattern_name, config in yaml_config.items():
            marker_json = _convert_marker_config_to_json(marker_pattern_name, config)
            marker_json_list.append(marker_json)
    return marker_json_list
