"""PPAL-based sample ordering for the mmplatform FiftyOne plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fiftyone as fo

from .kcenter import DEFAULT_SELECT, KCenterProgress

PPAL_RANK_FIELD = "ppal_rank"
DEFAULT_BUDGET_EXPAND_RATIO = 4
DEFAULT_LABEL_FIELD = "ground_truth"

_PLUGIN_DIR = Path(__file__).resolve().parent
_MMPLATFORM_ROOT = _PLUGIN_DIR.parent.parent.parent.parent
PPAL_ROOT = _MMPLATFORM_ROOT / "third_party" / "PPAL"
PPAL_WORK_DIRS = PPAL_ROOT / "work_dirs"
PPAL_RUNNER = PPAL_ROOT / "tools" / "fo_ppal_runner.py"

# COCO category ids for optional export of existing FiftyOne labels.
_COCO_NAME_TO_ID: Dict[str, int] = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


@dataclass
class PpalOrderResult:
    ordered_ids: List[str]
    checkpoint: str
    pool_size: int


def ppal_root() -> Path:
    return PPAL_ROOT


def ppal_work_dirs() -> Path:
    return PPAL_WORK_DIRS


def _checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    if name == "latest.pth":
        return (0, str(path.parent))
    if name.startswith("epoch_"):
        return (1, str(path))
    return (2, str(path))


def list_ppal_checkpoint_options() -> List[Dict[str, str]]:
    """Scan third_party/PPAL/work_dirs for PPAL-trained checkpoints."""
    root = ppal_work_dirs()
    if not root.is_dir():
        return []

    seen: set[str] = set()
    options: List[Dict[str, str]] = []
    paths = sorted(root.rglob("*.pth"), key=_checkpoint_sort_key)

    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        try:
            validate_ppal_checkpoint(resolved)
        except (ValueError, OSError):
            continue
        seen.add(resolved)
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        options.append({"name": resolved, "label": label})

    return options


def validate_ppal_checkpoint(checkpoint: str) -> str:
    """Require bbox_head.class_quality (PPAL-trained checkpoint only)."""
    import torch

    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint が見つかりません: {checkpoint}")

    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "bbox_head.class_quality" not in state:
        raise ValueError(
            "PPAL 学習済み checkpoint が必要です（bbox_head.class_quality がありません）。"
            f" third_party/PPAL/work_dirs 配下の学習結果を指定してください: {path}"
        )
    return str(path)


def normalize_ppal_checkpoint(checkpoint: Optional[str]) -> str:
    name = (checkpoint or "").strip()
    if not name:
        options = list_ppal_checkpoint_options()
        if not options:
            raise ValueError(
                "PPAL checkpoint がありません。"
                " third_party/PPAL/work_dirs に学習済み .pth を置いてください。"
            )
        return options[0]["name"]
    return validate_ppal_checkpoint(name)


def ensure_ppal_rank_field(dataset: fo.Dataset) -> None:
    if PPAL_RANK_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(PPAL_RANK_FIELD, fo.IntField)


def set_ppal_ranks(dataset: fo.Dataset, ordered_ids: Sequence[str]) -> None:
    ensure_ppal_rank_field(dataset)
    rank_map = {sample_id: rank for rank, sample_id in enumerate(ordered_ids)}
    dataset.set_values(PPAL_RANK_FIELD, rank_map, key_field="id")


def _resolve_ppal_python() -> str:
    if os.environ.get("PPAL_PYTHON"):
        return os.environ["PPAL_PYTHON"]
    venv_python = ppal_root() / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def _build_runner_env() -> dict:
    env = os.environ.copy()
    ppal = str(ppal_root().resolve())
    env["PYTHONPATH"] = ppal + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _parse_progress_line(line: str, progress: Optional[KCenterProgress]) -> None:
    if not line.startswith("PROGRESS:") or progress is None:
        return
    try:
        _, fraction, label = line.split(":", 2)
        progress.set(float(fraction), label)
    except ValueError:
        return


def _parse_runner_payload(stdout: str) -> Dict[str, Any]:
    """Parse JSON from runner stdout (may include mmcv logs before the payload)."""
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("PPAL runner が空の出力を返しました")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # mmcv progress bars can append JSON to the same line without a newline.
    start = text.rfind('{"')
    if start < 0:
        start = text.rfind("{")
    if start < 0:
        snippet = text[-500:] if len(text) > 500 else text
        raise RuntimeError(
            "PPAL runner の出力に JSON が含まれていません。"
            f" 末尾: {snippet!r}"
        )

    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as exc:
        snippet = text[start : start + 200]
        raise RuntimeError(
            f"PPAL runner の JSON を解釈できません: {exc}. 先頭: {snippet!r}"
        ) from exc


def _sample_needs_metadata(sample: fo.Sample) -> bool:
    if sample.metadata is None:
        return True
    width = getattr(sample.metadata, "width", None)
    height = getattr(sample.metadata, "height", None)
    return not width or not height


def _ensure_view_metadata(
    view: fo.core.view.DatasetView,
    progress: Optional[KCenterProgress] = None,
) -> None:
    missing_ids = [sample.id for sample in view if _sample_needs_metadata(sample)]
    if not missing_ids:
        return

    if progress is not None:
        progress.log(f"metadata を自動計算中… ({len(missing_ids)} 枚)")
        progress.set(0.03, f"metadata 計算中… ({len(missing_ids)} 枚)")

    view.dataset.select(missing_ids).compute_metadata()


def _detections_to_coco_boxes(
    sample: fo.Sample,
    label_field: str,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    detections_field = getattr(sample, label_field, None)
    if detections_field is None:
        return []

    detections = getattr(detections_field, "detections", None) or []
    exported: List[Dict[str, Any]] = []
    for det in detections:
        label = str(getattr(det, "label", "") or "").strip()
        if label not in _COCO_NAME_TO_ID:
            continue
        bbox = getattr(det, "bounding_box", None)
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = (float(v) for v in bbox)
        exported.append(
            {
                "label": label,
                "bbox": [
                    x * width,
                    y * height,
                    bw * width,
                    bh * height,
                ],
            }
        )
    return exported


def _collect_sample_records(
    view: fo.core.view.DatasetView,
    label_field: str = DEFAULT_LABEL_FIELD,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for sample in view:
        filepath = sample.filepath
        if not filepath:
            raise RuntimeError(f"画像パスが空です: sample id={sample.id}")

        width = getattr(sample.metadata, "width", None) if sample.metadata else None
        height = getattr(sample.metadata, "height", None) if sample.metadata else None
        if not width or not height:
            raise RuntimeError(
                f"metadata を取得できませんでした: {filepath}"
            )

        record: Dict[str, object] = {
            "fiftyone_id": sample.id,
            "filepath": filepath,
            "width": int(width),
            "height": int(height),
        }
        detections = _detections_to_coco_boxes(
            sample, label_field, int(width), int(height)
        )
        if detections:
            record["detections"] = detections
        records.append(record)
    return records


def compute_ppal_order(
    view: fo.core.view.DatasetView,
    checkpoint: str,
    *,
    budget: int = DEFAULT_SELECT,
    budget_expand_ratio: int = DEFAULT_BUDGET_EXPAND_RATIO,
    label_field: str = DEFAULT_LABEL_FIELD,
    progress: Optional[KCenterProgress] = None,
) -> PpalOrderResult:
    """Run PPAL selection via third_party/PPAL/tools/fo_ppal_runner.py."""
    checkpoint = validate_ppal_checkpoint(checkpoint)
    if not PPAL_RUNNER.is_file():
        raise FileNotFoundError(f"PPAL runner が見つかりません: {PPAL_RUNNER}")

    _ensure_view_metadata(view, progress=progress)
    sample_records = _collect_sample_records(view, label_field=label_field)
    if not sample_records:
        return PpalOrderResult(ordered_ids=[], checkpoint=checkpoint, pool_size=0)

    if progress is not None:
        progress.set(0.05, "PPAL runner を起動中…", force=True)
        progress.log(f"checkpoint: {checkpoint}")
        progress.log(f"対象サンプル数: {len(sample_records)}")
        progress.log("モード: image-only（COCO oracle 不要）")

    spec: Dict[str, object] = {
        "checkpoint": checkpoint,
        "samples": sample_records,
        "budget": int(budget),
        "budget_expand_ratio": int(budget_expand_ratio),
        "image_only": True,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(spec, tmp)
        spec_path = tmp.name

    cmd = [_resolve_ppal_python(), str(PPAL_RUNNER), spec_path]
    if progress is not None:
        progress.log(f"runner: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ppal_root()),
            env=_build_runner_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip("\n")
            if line:
                _parse_progress_line(line, progress)
                if progress is not None and not line.startswith("PROGRESS:"):
                    progress.log(line)

        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            detail = (stdout or "").strip()
            raise RuntimeError(
                f"PPAL runner が失敗しました (exit {proc.returncode}): {detail}"
            )

        payload = _parse_runner_payload(stdout)
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))

        ordered_ids = [str(x) for x in payload.get("ordered_fiftyone_ids", [])]
        for line in payload.get("log", []):
            if progress is not None:
                progress.log(str(line))

        if len(ordered_ids) != len(sample_records):
            raise RuntimeError(
                "PPAL の順位付け結果とサンプル数が一致しません "
                f"({len(ordered_ids)} != {len(sample_records)})"
            )

        if progress is not None:
            progress.set(0.97, "順位を保存中…")
            progress.log(f"{PPAL_RANK_FIELD} フィールドへ順位を保存")

        return PpalOrderResult(
            ordered_ids=ordered_ids,
            checkpoint=checkpoint,
            pool_size=int(payload.get("pool_size", 0)),
        )
    finally:
        try:
            os.unlink(spec_path)
        except OSError:
            pass
