from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetWriter:
    def __init__(self, output: str, manifest: dict[str, Any]) -> None:
        self.path = Path(output)
        self.manifest_path = self.path.with_suffix(".manifest.json")
        if self.path.exists() or self.manifest_path.exists():
            raise FileExistsError(f"Output gia' esistente: {self.path} o {self.manifest_path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        self._handle = self.path.open("x", encoding="utf-8", buffering=1)
        self.records = 0

    def append(self, record: dict[str, Any]) -> None:
        json.dump(record, self._handle, ensure_ascii=False, separators=(",", ":"))
        self._handle.write("\n")
        self._handle.flush()
        self.records += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

