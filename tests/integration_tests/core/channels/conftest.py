import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--voice-files",
        action="store",
        default=None,
        help="Path to a voice file (can also use VOICE_FILES env var)",
    )


@pytest.fixture
def voice_file(request):
    # prefer CLI option, fall back to env var
    val = request.config.getoption("--voice-files") or os.environ.get("VOICE_FILES")
    if not val:
        pytest.skip("no voice file provided via --voice-files or VOICE_FILES env var")

    # tolerate accidental `VOICE_FILES=...` strings
    if val.startswith("VOICE_FILES="):
        val = val.split("=", 1)[1]

    return os.path.normpath(val)
