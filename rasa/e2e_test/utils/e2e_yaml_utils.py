from datetime import datetime
from pathlib import Path
from typing import Any

import ruamel
import structlog

from rasa.e2e_test.constants import KEY_TEST_CASES

structlogger = structlog.get_logger()

DEFAULT_E2E_OUTPUT_TESTS_DIRECTORY = "e2e_tests"


class E2ETestYAMLWriter:
    def __init__(
        self,
        output_path: str = DEFAULT_E2E_OUTPUT_TESTS_DIRECTORY,
        **kwargs: Any,
    ) -> None:
        """Initializes the E2ETestYAMLWriter with necessary parameters.

        Args:
            output (str): Directory to save the generated tests.
        """
        self.output_path = output_path

    def write_to_file(self, tests: str) -> None:
        """Writes the provided test cases to a YAML file
        in the specified output directory.

        Args:
            tests (str): string containing the generated test cases.
        """
        if not tests.strip():
            structlogger.info("e2e_test_generator.no_tests_provided")
            return

        output_dir = Path(self.output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"e2e_tests_{timestamp}.yml"

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        yaml_data = ruamel.yaml.safe_load(tests)

        test_cases_yaml = [{KEY_TEST_CASES: yaml_data}]
        with open(output_file, "w") as outfile:
            yaml = ruamel.yaml.YAML()
            yaml.dump(test_cases_yaml, outfile)

        structlogger.info(
            "e2e_test_generator.tests_written_to_yaml", output_file=output_file
        )
