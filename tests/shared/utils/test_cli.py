import pytest

import rasa.shared.utils.cli


def test_print_error_and_exit():
    with pytest.raises(SystemExit):
        rasa.shared.utils.cli.print_error_and_exit("")
