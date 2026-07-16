"""Upload unlabeled images into data/unlabeled pools (COCO JSON, no bboxes)."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
UNLABELED_DIR = ROOT / "data" / "unlabeled"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_POOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_ANN_BASENAME = "instances_unlabeled.json"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def unlabeled_root() -> Path:
    rel = _env("MMPLATFORM_UNLABELED_DIR", "data/unlabeled")
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return path


def normalize_pool_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        raise ValueError("プール名を入力してください")
    if not _POOL_NAME_RE.fullmatch(value):
        raise ValueError(
            "プール名は 1–63 文字、英数字・_・- のみ（先頭は英数字）で指定してください"
        )
    return value


def pool_dir_basename(pool_name: str) -> str:
    stamp = datetime.now().strftime("%Y_%m%d_%H%M")
    return f"{normalize_pool_name(pool_name)}_{stamp}"


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


def _read_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow がインストールされていません。webui 環境で pip install Pillow してください。"
        ) from exc
    with Image.open(path) as im:
        width, height = im.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {path}")
    return width, height


def _write_coco_unlabeled(images_dir: Path, out_path: Path) -> int:
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and _is_image_filename(p.name)
    )
    images: List[dict] = []
    for index, path in enumerate(image_paths, start=1):
        width, height = _read_image_size(path)
        images.append(
            {
                "id": index,
                "file_name": path.name,
                "width": width,
                "height": height,
            }
        )
    payload = {
        "images": images,
        "annotations": [],
        "categories": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(images)


def upload_unlabeled_pool(
    pool_name: str,
    files: Sequence[Tuple[str, bytes]],
    tags: Optional[Iterable[str]] = None,
) -> dict:
    """Save images under data/unlabeled/<pool>_* and write instances_unlabeled.json."""
    if not files:
        raise ValueError("画像ファイルがありません")

    tag_list = [t.strip() for t in (tags or []) if t and t.strip()]
    root = unlabeled_root()
    root.mkdir(parents=True, exist_ok=True)

    basename = pool_dir_basename(pool_name)
    pool_dir = (root / basename).resolve()
    if pool_dir.parent != root.resolve():
        raise ValueError("プールの保存先が不正です")
    if pool_dir.exists():
        basename = f"{basename}_{uuid.uuid4().hex[:6]}"
        pool_dir = (root / basename).resolve()

    images_dir = pool_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    saved: List[Path] = []
    skipped: List[str] = []
    for filename, content in files:
        if not content:
            skipped.append(f"{filename} (empty)")
            continue
        if not _is_image_filename(filename):
            skipped.append(f"{filename} (unsupported type)")
            continue
        dest = _unique_path(images_dir, filename)
        dest.write_bytes(content)
        saved.append(dest.resolve())

    if not saved:
        if pool_dir.exists():
            for child in sorted(pool_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            pool_dir.rmdir()
        raise ValueError(
            "有効な画像ファイルがありません"
            + (f"; skipped: {', '.join(skipped[:5])}" if skipped else "")
        )

    ann_path = pool_dir / "annotations" / _ANN_BASENAME
    image_count = _write_coco_unlabeled(images_dir, ann_path)

    manifest = {
        "pool_id": basename,
        "pool_name": normalize_pool_name(pool_name),
        "tags": tag_list,
        "image_count": image_count,
        "ann_file": f"annotations/{_ANN_BASENAME}",
        "img_prefix": "images/",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (pool_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return {
        "pool_id": basename,
        "pool_name": manifest["pool_name"],
        "added": len(saved),
        "image_count": image_count,
        "skipped": skipped,
        "tags": tag_list,
        "data_root": str(pool_dir),
        "data_root_rel": str(pool_dir.relative_to(ROOT)),
        "ann_file": manifest["ann_file"],
        "img_prefix": manifest["img_prefix"],
        "timestamp": manifest["created_at"],
    }


def list_unlabeled_pools() -> List[dict]:
    """List pools under data/unlabeled that have instances_unlabeled.json."""
    root = unlabeled_root()
    if not root.is_dir():
        return []

    rows: List[dict] = []
    for pool_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not pool_dir.is_dir() or pool_dir.name.startswith("."):
            continue
        ann_path = pool_dir / "annotations" / _ANN_BASENAME
        images_dir = pool_dir / "images"
        if not ann_path.is_file() or not images_dir.is_dir():
            continue

        manifest: dict = {}
        manifest_path = pool_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}

        image_count = manifest.get("image_count")
        if image_count is None:
            try:
                coco = json.loads(ann_path.read_text(encoding="utf-8"))
                image_count = len(coco.get("images") or [])
            except json.JSONDecodeError:
                image_count = sum(
                    1
                    for p in images_dir.iterdir()
                    if p.is_file() and _is_image_filename(p.name)
                )

        rows.append(
            {
                "id": pool_dir.name,
                "pool_name": manifest.get("pool_name", pool_dir.name),
                "data_root_rel": str(pool_dir.relative_to(ROOT)),
                "ann_file": f"annotations/{_ANN_BASENAME}",
                "img_prefix": "images/",
                "image_count": image_count,
                "tags": manifest.get("tags") or [],
                "created_at": manifest.get("created_at"),
            }
        )
    return rows
