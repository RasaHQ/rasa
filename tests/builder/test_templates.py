import io
import tarfile
from pathlib import Path

import pytest

from rasa.cli.scaffold import ProjectTemplateName
from rasa.shared.constants import ASSISTANT_ID_DEFAULT_VALUE
from rasa.shared.utils.yaml import read_yaml


@pytest.mark.parametrize("template", ProjectTemplateName.get_all_values())
def test_template_use_the_default_id_in_config(template: ProjectTemplateName) -> None:
    # get the project folder
    project_folder = Path(f"rasa/cli/project_templates/{template}")
    # get the config file
    config_file = project_folder / "config.yml"
    # read the config file
    config = read_yaml(config_file)
    # check that the assistant_id is the default value
    assert config["assistant_id"] == ASSISTANT_ID_DEFAULT_VALUE


@pytest.mark.asyncio
async def test_download_cache_for_template_telco_creates_rasa_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # arrange
    project_folder = tmp_path / "proj"
    project_folder.mkdir(parents=True, exist_ok=True)

    # create an in-memory tar.gz containing a .rasa directory
    def _make_tar_with_rasa_dir() -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            dir_info = tarfile.TarInfo(".rasa")
            dir_info.type = tarfile.DIRTYPE
            tar.addfile(dir_info)

            # add a small file inside .rasa to ensure extraction creates the dir
            data = b"ok"
            file_info = tarfile.TarInfo(".rasa/foobar")
            file_info.size = len(data)
            tar.addfile(file_info, io.BytesIO(data))
        return buf.getvalue()

    tar_bytes = _make_tar_with_rasa_dir()

    # mock aiohttp session.get to return our tarball without real HTTP
    class _MockStream:
        def __init__(self, data: bytes) -> None:
            self._data = data

        async def iter_chunked(self, size: int):  # type: ignore[override]
            for i in range(0, len(self._data), size):
                yield self._data[i : i + size]

    class _MockResponse:
        def __init__(self, data: bytes) -> None:
            self.status = 200
            self.content = _MockStream(data)

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _MockSession:
        def __init__(self, data: bytes) -> None:
            self._data = data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str):
            return _MockResponse(self._data)

    # patch the ClientSession used inside download_cache_for_template
    monkeypatch.setattr(
        "rasa.builder.template_cache.aiohttp.ClientSession",
        lambda: _MockSession(tar_bytes),
    )

    # act
    from rasa.builder.template_cache import download_cache_for_template

    await download_cache_for_template(
        ProjectTemplateName.TELCO, project_folder.as_posix()
    )

    # assert
    assert (project_folder / ".rasa").exists()
