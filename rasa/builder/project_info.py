from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rasa.constants import RASA_DIR_NAME


@dataclass
class ProjectInfo:
    """Metadata persisted for the builder about the current project.

    - first_used_utc_iso: RFC3339/ISO8601 string of when the project was first used.
    """

    first_used_utc_iso: Optional[str] = None

    @classmethod
    def from_json(cls, data: str) -> "ProjectInfo":
        payload = json.loads(data) if data else {}
        return cls(**payload)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def first_used_dt(self) -> Optional[datetime]:
        if not self.first_used_utc_iso:
            return None
        value = self.first_used_utc_iso.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(value)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


def _project_info_path(project_folder: Path) -> Path:
    return project_folder / RASA_DIR_NAME / "project_info.json"


def load_project_info(project_folder: Path) -> ProjectInfo:
    path = _project_info_path(project_folder)
    try:
        if path.exists():
            return ProjectInfo.from_json(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return ProjectInfo()


def save_project_info(project_folder: Path, info: ProjectInfo) -> None:
    path = _project_info_path(project_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(info.to_json(), encoding="utf-8")


def ensure_first_used(project_folder: Path) -> datetime:
    info = load_project_info(project_folder)
    if existing := info.first_used_dt():
        return existing

    now = datetime.now(timezone.utc)
    info.first_used_utc_iso = now.isoformat()
    save_project_info(project_folder, info)
    return now
