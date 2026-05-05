from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"


def project_paths(project_root: Optional[str | Path] = None) -> ProjectPaths:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    return ProjectPaths(project_root=root)


def race_weekend_window(race_date: datetime) -> tuple[datetime, datetime]:
    """
    Define a consistent time window for weather enrichment.

    This is intentionally simple and will be refined once the exact session timestamps
    are selected (practice/quali/race).
    """
    start = race_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(hour=23, minute=59, second=59, microsecond=0)
    return start, end

