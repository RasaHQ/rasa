from typing import Any, Dict, List, Text, Union

import pytest

from rasa.cli.e2e_test import read_e2e_test_schema


@pytest.fixture(scope="module")
def e2e_schema() -> Union[List[Any], Dict[Text, Any]]:
    return read_e2e_test_schema()
