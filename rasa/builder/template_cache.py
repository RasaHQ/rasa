import os
from pathlib import Path

import aiofiles
import aiofiles.os
import aioshutil
import structlog

from rasa.cli.scaffold import ProjectTemplateName

structlogger = structlog.get_logger()


# Root directory for storing downloaded template caches on disk.
_CACHE_ROOT_DIR = Path(os.getenv("RASA_TEMPLATE_CACHE_DIR", "/templates"))


def _template_cache_dir(template: ProjectTemplateName) -> Path:
    """Return the local cache directory for a given template and version."""
    return _CACHE_ROOT_DIR / template.value


async def _copytree(src: Path, dst: Path) -> None:
    """Copy directory tree from src to dst, merging into dst.

    Existing files are overwritten. Hidden files and directories are included, as
    caches can contain `.rasa` metadata that should be applied before calling
    `ensure_first_used`.
    """
    for root, dirs, files in os.walk(src):
        rel_path = Path(root).relative_to(src)
        target_dir = dst / rel_path
        await aiofiles.os.makedirs(target_dir, exist_ok=True)
        for filename in files:
            src_file = Path(root) / filename
            dst_file = target_dir / filename
            await aioshutil.copy2(src_file, dst_file)


async def copy_cache_for_template_if_available(
    template: ProjectTemplateName, project_folder: Path
) -> None:
    """Copy a previously downloaded cache for `template` into `project_folder`.

    If the cache does not exist, this function is a no-op.
    """
    try:
        cache_dir = _template_cache_dir(template)
        if cache_dir.exists() and any(cache_dir.iterdir()):
            await _copytree(cache_dir, project_folder)
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
