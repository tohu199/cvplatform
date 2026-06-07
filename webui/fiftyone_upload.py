"""Upload local image files into a FiftyOne dataset via the web UI."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")


def _env(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default).strip() or default


def default_dataset_name() -> str:
    return _env("MMPLATFORM_FIFTYONE_DATASET", "mmplatform")


def fiftyone_app_url() -> str:
    return _env("MMPLATFORM_FIFTYONE_APP_URL", "http://localhost:5151")


def upload_root() -> Path:
    rel = _env("MMPLATFORM_FIFTYONE_UPLOAD_DIR", "data/fiftyone/uploads")
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return path


def normalize_dataset_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        raise ValueError("dataset name is required")
    if not _DATASET_NAME_RE.fullmatch(value):
        raise ValueError(
            "dataset name must be 1–63 chars: letters, digits, '_', '-' "
            "(must start with letter or digit)"
        )
    return value


def dataset_upload_dir(dataset_name: str) -> Path:
    return upload_root() / normalize_dataset_name(dataset_name)


def _require_fiftyone():
    try:
        import fiftyone as fo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "fiftyone is not installed in this Python environment. "
            "Install it (e.g. pip install fiftyone) and restart the webui server."
        ) from exc


def list_datasets() -> List[str]:
    _require_fiftyone()
    import fiftyone as fo

    return sorted(fo.list_datasets())


def _safe_basename(filename: str) -> str:
    base = Path(filename or "upload").name
    if base in ("", ".", ".."):
        raise ValueError(f"invalid filename: {filename!r}")
    return base


def _unique_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base = _safe_basename(filename)
    dest = directory / base
    if not dest.exists():
        return dest
    stem = Path(base).stem
    suffix = Path(base).suffix
    for index in range(1, 10000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError(f"could not allocate a unique path for {filename!r}")


def _is_image_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_SUFFIXES


def upload_images(
    dataset_name: str,
    files: Sequence[Tuple[str, bytes]],
    tags: Optional[Iterable[str]] = None,
) -> dict:
    """Save *files* to disk and register them as samples in a FiftyOne dataset."""
    _require_fiftyone()
    import fiftyone as fo

    name = normalize_dataset_name(dataset_name)
    if not files:
        raise ValueError("no files provided")

    tag_list = [t.strip() for t in (tags or []) if t and t.strip()]
    upload_dir = dataset_upload_dir(name)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    skipped: List[str] = []
    for filename, content in files:
        if not content:
            skipped.append(f"{filename} (empty)")
            continue
        if not _is_image_filename(filename):
            skipped.append(f"{filename} (unsupported type)")
            continue
        dest = _unique_path(upload_dir, filename)
        dest.write_bytes(content)
        saved.append(dest.resolve())

    if not saved:
        raise ValueError(
            "no valid image files to import"
            + (f"; skipped: {', '.join(skipped[:5])}" if skipped else "")
        )

    if name in fo.list_datasets():
        dataset = fo.load_dataset(name)
    else:
        dataset = fo.Dataset(name)

    samples = [fo.Sample(filepath=str(path), tags=tag_list or None) for path in saved]
    dataset.add_samples(samples)

    return {
        "dataset": name,
        "added": len(saved),
        "skipped": skipped,
        "upload_dir": str(upload_dir.resolve()),
        "sample_count": dataset.count(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
