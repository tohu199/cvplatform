from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import yaml
from cvat_sdk import Client
from cvat_sdk.core.proxies.types import Location

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGIN_INFO_PATH = REPO_ROOT / "login_info.yaml"
DEFAULT_CVAT_HOST = "http://localhost:8080"
FORMAT_NAME = "COCO 1.0"


def export_date_stamp(when: Optional[datetime] = None) -> str:
    """Timestamp for export filenames, e.g. ``2026_0514_0121`` (YYYY_MMDD_HHMM)."""
    dt = when or datetime.now()
    return dt.strftime("%Y_%m%d_%H%M")


def default_export_zip(task_id: int, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or REPO_ROOT
    name = f"task_{task_id}_{export_date_stamp()}.zip"
    return root / "data" / "exports" / name


def _load_login_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise SystemExit(
            f"{path}: top-level YAML must be a mapping (dict), not {type(config).__name__}"
        )
    return config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a CVAT task dataset (COCO) and unzip locally.")
    p.add_argument(
        "--task-id",
        type=int,
        default=1,
        metavar="ID",
        help="CVAT task id to export (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="ZIP",
        help=(
            "Output zip path (default: <repo>/data/exports/task_<task-id>_<YYYY_MMDD_HHMM>.zip)"
        ),
    )
    args = p.parse_args()
    if args.out is None:
        args.out = default_export_zip(args.task_id)
    else:
        args.out = args.out.expanduser().resolve()
    return args


def run_export(task_id: int, out_zip: Optional[Path] = None) -> Tuple[Path, Path]:
    """Export CVAT task to *out_zip* (default: :func:`default_export_zip`) and unzip next to it.

    Returns ``(out_zip, extract_dir)``.
    """
    if out_zip is None:
        out_zip = default_export_zip(task_id)
    else:
        out_zip = out_zip.expanduser().resolve()

    extract_dir = out_zip.parent / out_zip.stem

    creds = _load_login_yaml(_LOGIN_INFO_PATH)
    username = creds.get("USERNAME")
    password = creds.get("PASSWORD")
    if not username or not password:
        raise ValueError(
            f"USERNAME and PASSWORD must be set in {_LOGIN_INFO_PATH} (YAML mapping)."
        )
    cvat_host = creds.get("HOST") or DEFAULT_CVAT_HOST

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with Client(cvat_host) as client:
        client.login((username, password))
        task = client.tasks.retrieve(task_id)
        task.export_dataset(
            format_name=FORMAT_NAME,
            filename=out_zip,
            include_images=True,
            location=Location.LOCAL,
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "r") as zf:
        zf.extractall(extract_dir)

    return out_zip, extract_dir


def main() -> None:
    args = _parse_args()
    try:
        out_zip, extract_dir = run_export(args.task_id, args.out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(out_zip)
    print(extract_dir)


if __name__ == "__main__":
    main()
