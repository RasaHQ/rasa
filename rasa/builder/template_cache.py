import asyncio
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Generator

import aiofiles
import aiohttp
import structlog
from sanic import Sanic

import rasa.version
from rasa.builder.logging_utils import capture_exception_with_context
from rasa.cli.scaffold import ProjectTemplateName

structlogger = structlog.get_logger()

CACHE_BUCKET_URL = "https://trained-templates.s3.us-east-1.amazonaws.com"

# Root directory for storing downloaded template caches on disk.
_CACHE_ROOT_DIR = Path(
    os.getenv(
        "RASA_TEMPLATE_CACHE_DIR",
        Path.home().joinpath(".rasa", "template-cache").as_posix(),
    )
)


def _template_cache_dir(template: ProjectTemplateName) -> Path:
    """Return the local cache directory for a given template and version."""
    return _CACHE_ROOT_DIR / rasa.version.__version__ / template.value


def _cache_root_dir() -> Path:
    return Path(
        os.getenv(
            "RASA_TEMPLATE_CACHE_DIR",
            Path.home().joinpath(".rasa", "template-cache").as_posix(),
        )
    )


def _safe_tar_members(
    tar: tarfile.TarFile, destination_directory: Path
) -> Generator[tarfile.TarInfo, None, None]:
    """Yield safe members for extraction to prevent path traversal and links.

    Args:
        tar: Open tar file handle
        destination_directory: Directory to which files will be extracted

    Yields:
        Members that are safe to extract within destination_directory
    """
    base_path = destination_directory.resolve()

    for member in tar.getmembers():
        name = member.name
        # Skip empty names and absolute paths
        if not name or name.startswith("/") or name.startswith("\\"):
            continue

        # Disallow symlinks and hardlinks
        if member.issym() or member.islnk():
            continue

        # Compute the final path and ensure it's within base_path
        target_path = (base_path / name).resolve()
        try:
            target_path.relative_to(base_path)
        except ValueError:
            # Member would escape the destination directory
            continue

        yield member


def _copytree(src: Path, dst: Path) -> None:
    """Copy directory tree from src to dst, merging into dst.

    Existing files are overwritten. Hidden files and directories are included, as
    caches can contain `.rasa` metadata that should be applied before calling
    `ensure_first_used`.
    """
    for root, dirs, files in os.walk(src):
        rel_path = Path(root).relative_to(src)
        target_dir = dst / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            src_file = Path(root) / filename
            dst_file = target_dir / filename
            shutil.copy2(src_file, dst_file)


async def download_cache_for_template(
    template: ProjectTemplateName, target_dir: str
) -> None:
    # get a temp path for the cache file download
    temporary_cache_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)

    try:
        url = f"{CACHE_BUCKET_URL}/{rasa.version.__version__}-{template.value}.tar.gz"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                async with aiofiles.open(temporary_cache_file.name, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        await f.write(chunk)

        # extract the cache to the project folder using safe member filtering
        with tarfile.open(temporary_cache_file.name, "r:gz") as tar:
            destination = Path(target_dir)
            destination.mkdir(parents=True, exist_ok=True)
            tar.extractall(
                path=destination,
                members=_safe_tar_members(tar, destination),
            )

        structlogger.info(
            "project_generator.download_cache_for_template.success",
            template=template,
            event_info=(
                "Downloaded cache for template, extracted to target directory."
            ),
            target_dir=target_dir,
        )
    except aiohttp.ClientResponseError as e:
        if e.status == 403:
            structlogger.debug(
                "project_generator.download_cache_for_template.no_cache_found",
                template=template,
                event_info=("No cache found for template, continuing without it."),
                target_dir=target_dir,
            )
        else:
            capture_exception_with_context(
                e,
                "project_generator.download_cache_for_template.response_error",
                extra={
                    "template": template.value,
                    "status": str(e.status),
                    "target_dir": target_dir,
                },
            )
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "project_generator.download_cache_for_template.unexpected_error",
            extra={"template": template.value, "target_dir": target_dir},
        )
    finally:
        # Clean up the temporary file
        try:
            Path(temporary_cache_file.name).unlink(missing_ok=True)
        except Exception as exc:
            structlogger.debug(
                "project_generator.download_cache_for_template.cleanup_error",
                error=str(exc),
                template=template,
                event_info=("Failed to cleanup cache for template, ignoring."),
            )


async def background_download_template_caches(
    app: Sanic, loop: asyncio.AbstractEventLoop
) -> None:
    """Kick off background downloads of template caches if enabled."""
    try:
        structlogger.info(
            "builder.main.background_cache_download.start",
            event_info=(
                "Starting background download of template caches for this " "version"
            ),
        )

        # Ensure cache root exists
        _cache_root_dir().mkdir(parents=True, exist_ok=True)

        async def _download(template: ProjectTemplateName) -> None:
            try:
                target_dir = _template_cache_dir(template)
                if target_dir.exists() and any(target_dir.iterdir()):
                    structlogger.debug(
                        "builder.main.background_cache_download.skipped",
                        template=template,
                        event_info=(
                            "Skipping download of template cache because it "
                            "already exists."
                        ),
                        target_dir=target_dir,
                    )
                    return

                target_dir.mkdir(parents=True, exist_ok=True)
                await download_cache_for_template(template, target_dir.as_posix())
            except Exception as exc:
                structlogger.debug(
                    "builder.main.background_cache_download.error",
                    template=template,
                    error=str(exc),
                )

        # schedule downloads concurrently without blocking startup
        for template in ProjectTemplateName:
            loop.create_task(_download(template))
    except Exception as exc:
        structlogger.debug(
            "builder.main.background_cache_download.unexpected_error",
            error=str(exc),
        )


def copy_cache_for_template_if_available(
    template: ProjectTemplateName, project_folder: Path
) -> None:
    """Copy a previously downloaded cache for `template` into `project_folder`.

    If the cache does not exist, this function is a no-op.
    """
    try:
        cache_dir = _template_cache_dir(template)
        if cache_dir.exists() and any(cache_dir.iterdir()):
            _copytree(cache_dir, project_folder)
            structlogger.info(
                "project_generator.copy_cache_for_template.success",
                template=template,
                event_info=(
                    "Copied cached template files from cache to project folder."
                ),
            )
        else:
            structlogger.debug(
                "project_generator.copy_cache_for_template.missing",
                template=template,
                event_info=("No local cache found for template; skipping copy."),
            )
    except Exception as exc:
        structlogger.warning(
            "project_generator.copy_cache_for_template.error",
            error=str(exc),
            template=template,
        )
