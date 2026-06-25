"""Run PPAL RetinaNet training from a JSON spec (subprocess entry for the Web UI)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from webui.ppal_defaults import DEFAULT_PPAL_CONFIG_REL


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ppal_root() -> Path:
    return _repo_root() / "third_party" / "PPAL"


def _resolve_ppal_python() -> str:
    venv_python = _ppal_root() / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def _prepare_coco_json(
    ann_path: Path,
    out_path: Path,
    class_names: Optional[List[str]],
    *,
    require_annotations: bool,
) -> Path:
    from webui.coco_merge import filter_coco_by_class_names

    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    if class_names:
        raw = filter_coco_by_class_names(raw, class_names)
    if require_annotations and not raw.get("annotations"):
        raise ValueError(f"アノテーションが 0 件です: {ann_path}")
    if not raw.get("images"):
        raise ValueError(f"画像が 0 件です: {ann_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw), encoding="utf-8")
    return out_path


def _prepare_unlabeled_json(categories: List[dict], work_dir: Path) -> Path:
    out = work_dir / "unlabeled.json"
    payload = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _resolve_img_prefix(data_root: Path, img_prefix: str) -> str:
    rel = img_prefix.strip().replace("\\", "/").strip("/")
    prefix = str((data_root / rel).resolve())
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _legacy_sources(spec: dict, key: str) -> List[Dict[str, str]]:
    dr = spec.get(f"{key}_data_root")
    ann = spec.get(f"{key}_ann_file")
    prefix = spec.get(f"{key}_img_prefix")
    if not dr or not ann or not prefix:
        return []
    return [{"data_root": dr, "ann_file": ann, "img_prefix": prefix}]


def _sources_from_spec(spec: dict, key: str) -> List[Dict[str, str]]:
    rows = spec.get(f"{key}_sources")
    if isinstance(rows, list) and rows:
        return rows
    return _legacy_sources(spec, key)


def _prepare_labeled_sources(
    sources: List[Dict[str, str]],
    work_dir: Path,
    classes: Optional[List[str]],
) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    categories: List[dict] = []
    for i, source in enumerate(sources):
        data_root = Path(source["data_root"]).resolve()
        ann_path = data_root / source["ann_file"]
        if not ann_path.is_file():
            raise FileNotFoundError(f"labeled アノテーションが見つかりません: {ann_path}")
        out = work_dir / "sources" / f"labeled_{i}.json"
        _prepare_coco_json(ann_path, out, classes, require_annotations=True)
        if not categories:
            raw = json.loads(out.read_text(encoding="utf-8"))
            categories = raw.get("categories") or []
        prepared.append(
            {
                "ann_file": str(out.resolve()),
                "img_prefix": _resolve_img_prefix(data_root, source["img_prefix"]),
            }
        )
    return prepared


def _prepare_val_sources(
    sources: List[Dict[str, str]],
    work_dir: Path,
    classes: Optional[List[str]],
) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for i, source in enumerate(sources):
        data_root = Path(source["data_root"]).resolve()
        ann_path = data_root / source["ann_file"]
        if not ann_path.is_file():
            raise FileNotFoundError(f"val アノテーションが見つかりません: {ann_path}")
        out = work_dir / "sources" / f"val_{i}.json"
        _prepare_coco_json(ann_path, out, classes, require_annotations=True)
        prepared.append(
            {
                "ann_file": str(out.resolve()),
                "img_prefix": _resolve_img_prefix(data_root, source["img_prefix"]),
            }
        )
    return prepared


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m webui.ppal_train_worker <spec.json>", flush=True)
        return 2

    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    ppal_root = _ppal_root()
    if not ppal_root.is_dir():
        raise FileNotFoundError(f"PPAL が見つかりません: {ppal_root}")

    work_dir = Path(spec["work_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    labeled_sources = _sources_from_spec(spec, "labeled")
    val_sources = _sources_from_spec(spec, "val")
    if not labeled_sources:
        raise ValueError("labeled_sources が空です。")
    if not val_sources:
        raise ValueError("val_sources が空です。")

    classes = spec.get("classes")
    labeled_prepared = _prepare_labeled_sources(labeled_sources, work_dir, classes)
    val_prepared = _prepare_val_sources(val_sources, work_dir, classes)

    first_labeled = json.loads(Path(labeled_prepared[0]["ann_file"]).read_text(encoding="utf-8"))
    unlabeled = _prepare_unlabeled_json(
        first_labeled.get("categories") or [], work_dir
    )

    launch_spec: Dict[str, Any] = {
        "config_path": spec.get("config_path") or DEFAULT_PPAL_CONFIG_REL,
        "work_dir": str(work_dir),
        "classes": classes,
        "max_epochs": spec.get("max_epochs"),
        "lr": spec.get("lr"),
        "batch_size": spec.get("batch_size"),
        "labeled_sources": labeled_sources,
        "val_sources": val_sources,
        "labeled_prepared": labeled_prepared,
        "val_prepared": val_prepared,
        "labeled_data": str((work_dir / "sources" / "labeled_0.json").resolve()),
        "unlabeled_data": str(unlabeled.resolve()),
    }
    if spec.get("resume_from"):
        launch_spec["resume_from"] = spec["resume_from"]

    launch_spec_path = work_dir / "webui_ppal_launch_spec.json"
    launch_spec_path.write_text(json.dumps(launch_spec, indent=2), encoding="utf-8")

    cmd = [
        _resolve_ppal_python(),
        "-u",
        "-m",
        "webui.ppal_train_launch",
        str(launch_spec_path),
    ]
    print("PPAL launch command:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    root = str(_repo_root())
    ppal = str(ppal_root)
    env["PYTHONPATH"] = root + os.pathsep + ppal + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, cwd=str(ppal_root), env=env, check=False)
    return proc.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
