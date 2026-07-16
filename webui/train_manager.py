"""Training job lifecycle: discover exports, spawn MMDet worker, parse logs."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, TypedDict

ROOT = Path(__file__).resolve().parent.parent

from webui.nuclio_deploy import (
    WORK_DIR_NAME_RE,
    WORK_DIR_PARENT,
    _resolve_checkpoint_path,
    resolve_work_dir,
)
from webui.coco_categories import suggest_categories, validate_selected_classes
from webui.coco_merge import (
    build_categories_for_class_names,
    prepare_unlabeled_ann_files,
    prepare_val_evaluator_ann_file,
)
from webui.unlabeled_upload import list_unlabeled_pools, unlabeled_root
from webui.work_dir_summary import find_scalars_json
EXPORTS_DIR = ROOT / "data" / "exports"
MMDET_ROOT = ROOT / "third_party" / "mmdetection"
DEFAULT_CONFIG_REL = "configs/yolox/yolox_s_finetune.py"
DEFAULT_SEMI_CONFIG_REL = "configs/soft_teacher/soft-teacher_faster-rcnn_finetune.py"
DEFAULT_CONFIG = MMDET_ROOT / DEFAULT_CONFIG_REL.replace("/", os.sep)

SEMI_MODEL_TYPES = frozenset({"SoftTeacher", "SemiBaseDetector"})

# 半教師あり（Soft-Teacher）向け UI 既定値
DEFAULT_SEMI_MAX_ITERS = 10000
DEFAULT_SEMI_LR = 0.001
DEFAULT_SEMI_BATCH_SIZE = 5
DEFAULT_SEMI_VAL_INTERVAL = 2000

# 教師あり向け UI 既定値（yolox_s_finetune.py の interval=5 に合わせる）
DEFAULT_VAL_INTERVAL_EPOCHS = 5

EPOCH_TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\]"
)
LOSS_TOKEN_RE = re.compile(r"\bloss:\s+([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
WEB_TRAIN_PREFIX = "web_train_"

# チャート初期表示（フロントと揃える）
DEFAULT_CHART_METRICS = ["loss", "memory", "coco/bbox_mAP"]
DEFAULT_SEMI_CHART_METRICS = [
    "loss",
    "memory",
    "student/coco/bbox_mAP",
    "teacher/coco/bbox_mAP",
]

_VAL_MAP_MARKERS = (
    "coco/bbox_mAP",
    "student/coco/bbox_mAP",
    "teacher/coco/bbox_mAP",
)


def export_stamp() -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M")


def default_work_dir_basename() -> str:
    return f"web_train_{export_stamp()}_{uuid.uuid4().hex[:8]}"


def normalize_work_dir_basename(name: Optional[str]) -> str:
    """work_dirs 直下のフォルダ名を web_train_* 形式に正規化する。"""
    if not name or not name.strip():
        return default_work_dir_basename()
    n = name.strip()
    if n in (".", "..") or "/" in n or "\\" in n:
        raise ValueError("work_dir 名にパス区切りは使えません。")
    full = n if n.startswith(WEB_TRAIN_PREFIX) else f"{WEB_TRAIN_PREFIX}{n}"
    if len(full) > 120:
        raise ValueError("work_dir 名は 120 文字以内にしてください。")
    if not WORK_DIR_NAME_RE.match(full):
        raise ValueError(
            "work_dir 名は web_train_ の後に英数字と _ - のみ使用できます（例: web_train_task5_from_task4）。"
        )
    return full


def allocate_work_dir(basename: Optional[str] = None) -> Path:
    """新規学習用 work_dir を確保。basename 未指定時は既定の web_train_* 名。"""
    name = normalize_work_dir_basename(basename)
    wd = (WORK_DIR_PARENT / name).resolve()
    if wd.parent != WORK_DIR_PARENT.resolve():
        raise ValueError("work_dir の配置が不正です。")
    if wd.exists():
        raise ValueError(f"work_dir が既に存在します: {name}")
    wd.mkdir(parents=True, exist_ok=False)
    return wd


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def list_pretrained_work_dirs() -> List[Dict[str, Any]]:
    """work_dirs 内で last_checkpoint から解決できる学習済みモデル一覧。"""
    rows: List[Dict[str, Any]] = []
    if not WORK_DIR_PARENT.is_dir():
        return rows
    for p in sorted(WORK_DIR_PARENT.iterdir(), key=lambda x: x.name, reverse=True):
        if not p.is_dir() or not p.name.startswith("web_train_"):
            continue
        if not WORK_DIR_NAME_RE.match(p.name):
            continue
        if not (p / "last_checkpoint").is_file():
            continue
        try:
            ckpt, ck_base = _resolve_checkpoint_path(p)
        except ValueError:
            continue
        rows.append(
            {
                "id": p.name,
                "checkpoint": ck_base,
                "checkpoint_path": str(ckpt),
            }
        )
    return rows


def is_semi_config_rel(config_rel: str) -> bool:
    norm = config_rel.replace("\\", "/")
    return "/soft_teacher/" in f"/{norm}/"


def train_defaults() -> Dict[str, Any]:
    return {
        "default_config": DEFAULT_CONFIG_REL,
        "default_semi_config": DEFAULT_SEMI_CONFIG_REL,
        "default_semi_metrics": list(DEFAULT_SEMI_CHART_METRICS),
        "semi_max_iters": DEFAULT_SEMI_MAX_ITERS,
        "semi_lr": DEFAULT_SEMI_LR,
        "semi_batch_size": DEFAULT_SEMI_BATCH_SIZE,
        "semi_val_interval": DEFAULT_SEMI_VAL_INTERVAL,
        "val_interval_epochs": DEFAULT_VAL_INTERVAL_EPOCHS,
    }


def _is_val_map_line(obj: Dict[str, Any]) -> bool:
    return any(k in obj for k in _VAL_MAP_MARKERS)


def _load_work_dir_semi_flag(work_dir: Optional[str]) -> bool:
    if not work_dir:
        return False
    spec_path = Path(work_dir) / "webui_train_spec.json"
    if not spec_path.is_file():
        return False
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(spec.get("semi"))


def resolve_default_chart_metrics(
    work_dir: Optional[str] = None,
    available_metrics: Optional[Sequence[str]] = None,
) -> List[str]:
    semi = _load_work_dir_semi_flag(work_dir)
    if not semi and available_metrics:
        semi = any(
            m.startswith("student/") or m.startswith("teacher/")
            for m in available_metrics
        )
    return list(DEFAULT_SEMI_CHART_METRICS if semi else DEFAULT_CHART_METRICS)


def _resolve_val_interval(
    val_interval: Optional[int],
    *,
    limit: int,
    default: int,
    label: str,
) -> int:
    value = default if val_interval is None else int(val_interval)
    if value < 1:
        raise ValueError(f"{label} は 1 以上にしてください。")
    if value > limit:
        raise ValueError(f"{label} は {limit} 以下にしてください。")
    return value


def list_unlabeled_pools_for_train() -> List[Dict[str, Any]]:
    return list_unlabeled_pools()


def resolve_mmdet_config_path(config_path: Optional[str]) -> Path:
    """third_party/mmdetection 配下の config パスを解決する。"""
    raw = (config_path or "").strip() or DEFAULT_CONFIG_REL
    norm = raw.replace("\\", "/")
    if Path(raw).is_absolute():
        cfg_path = Path(raw).resolve()
    elif norm.startswith("third_party/mmdetection/"):
        cfg_path = (ROOT / norm).resolve()
    elif norm.startswith("configs/"):
        cfg_path = (MMDET_ROOT / norm).resolve()
    else:
        cfg_path = (MMDET_ROOT / "configs" / norm.lstrip("/")).resolve()
    if not cfg_path.is_file():
        raise ValueError(f"config が見つかりません: {cfg_path}")
    if not _is_under(cfg_path, MMDET_ROOT.resolve()):
        raise ValueError("config は third_party/mmdetection 配下のファイルを指定してください。")
    return cfg_path


def list_mmdet_configs() -> List[Dict[str, Any]]:
    """MMDetection configs/*.py 一覧（_base_ 除く）。"""
    configs_dir = MMDET_ROOT / "configs"
    if not configs_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(configs_dir.rglob("*.py")):
        rel = p.relative_to(MMDET_ROOT).as_posix()
        if "/_base_/" in f"/{rel}/" or p.name == "__init__.py" or p.name.startswith("_"):
            continue
        parts = rel.split("/")
        group = parts[1] if len(parts) > 2 else "other"
        label = "/".join(parts[2:]) if len(parts) > 2 else p.name
        rows.append(
            {
                "id": rel,
                "label": label,
                "group": group,
                "is_default": rel == DEFAULT_CONFIG_REL,
                "is_semi": is_semi_config_rel(rel),
                "is_semi_default": rel == DEFAULT_SEMI_CONFIG_REL,
            }
        )
    rows.sort(key=lambda r: (not r["is_default"], r["group"], r["label"]))
    return rows


def list_export_datasets() -> List[Dict[str, Any]]:
    if not EXPORTS_DIR.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(EXPORTS_DIR.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == ".trash":
            continue
        ann_dir = p / "annotations"
        if not ann_dir.is_dir():
            continue
        jsons = sorted(
            f.relative_to(p).as_posix()
            for f in ann_dir.glob("instances*.json")
        )
        if not jsons:
            continue
        prefixes: List[str] = []
        im = p / "images"
        if im.is_dir():
            for sub in sorted(im.iterdir()):
                if sub.is_dir():
                    prefixes.append(f"images/{sub.name}/")
        if not prefixes:
            prefixes.append("images/")
        rows.append(
            {
                "id": p.name,
                "data_root_rel": str(p.relative_to(ROOT)),
                "annotations": jsons,
                "image_prefixes": prefixes,
            }
        )
    return rows


def _parse_train_loss_line(line: str) -> Optional[Dict[str, Any]]:
    m = EPOCH_TRAIN_RE.search(line)
    if not m:
        return None
    lm = LOSS_TOKEN_RE.search(line)
    if not lm:
        return None
    epoch = int(m.group(1))
    it = int(m.group(2))
    total = int(m.group(3))
    try:
        loss = float(lm.group(1))
    except ValueError:
        return None
    return {"epoch": epoch, "iter": it, "iter_total": total, "loss": loss}


def _metric_sort_key(name: str) -> tuple:
    if name == "loss":
        return (0, name)
    if name.startswith("loss_"):
        return (1, name)
    if name == "memory":
        return (2, name)
    if name == "student/coco/bbox_mAP":
        return (3, name)
    if name == "teacher/coco/bbox_mAP":
        return (4, name)
    if name.startswith("student/coco/bbox_mAP"):
        return (5, name)
    if name.startswith("teacher/coco/bbox_mAP"):
        return (6, name)
    if name.startswith("coco/bbox_mAP"):
        return (7, name)
    return (9, name)


def _parse_scalars_json(path: Path) -> tuple:
    series: Dict[str, List[Dict[str, Any]]] = {}
    keys_seen: set = set()
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, []
    last_train_step = 0.0
    val_event_idx = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        step = obj.get("step")
        if step is None:
            continue
        try:
            step_f = float(step)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(step_f):
            continue
        is_val = _is_val_map_line(obj) and "loss" not in obj
        if "loss" in obj and not is_val:
            last_train_step = step_f
        effective_step = step_f
        if is_val and step_f == 0:
            if last_train_step > 0:
                effective_step = last_train_step
            else:
                val_event_idx += 1
                effective_step = float(val_event_idx)
        for k, v in obj.items():
            if k == "step":
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            fv = float(v)
            if not math.isfinite(fv):
                continue
            keys_seen.add(k)
            series.setdefault(k, []).append({"step": effective_step, "value": fv})
    available = sorted(keys_seen, key=_metric_sort_key)
    return series, available


@dataclass
class TrainJob:
    job_id: str
    status: str  # idle starting running completed failed stopped
    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=12000))
    loss_points: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2000))
    error: Optional[str] = None
    work_dir: Optional[str] = None
    command: Optional[List[str]] = None
    _proc: Optional[subprocess.Popen] = None
    _reader_t: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loss_step: int = field(default=0, init=False, repr=False)

    def append_line(self, text: str) -> None:
        with self._lock:
            self.lines.append(text.rstrip("\n"))
            parsed = _parse_train_loss_line(text)
            if parsed is not None:
                self._loss_step += 1
                self.loss_points.append({"step": self._loss_step, **parsed})

    def snapshot(self, tail: int = 400) -> Dict[str, Any]:
        with self._lock:
            lst = list(self.lines)
            lp = list(self.loss_points)
            wd = self.work_dir
        if tail > 0 and len(lst) > tail:
            lst = lst[-tail:]
        metric_series: Dict[str, List[Dict[str, Any]]] = {}
        available_metrics: List[str] = []
        scalars_path: Optional[str] = None
        if wd:
            sp = find_scalars_json(Path(wd))
            if sp is not None and sp.is_file():
                metric_series, available_metrics = _parse_scalars_json(sp)
                scalars_path = str(sp)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "work_dir": self.work_dir,
            "command": self.command,
            "error": self.error,
            "lines_tail": lst,
            "loss_points": lp,
            "metric_series": metric_series,
            "available_metrics": available_metrics,
            "scalars_path": scalars_path,
            "default_metrics": resolve_default_chart_metrics(
                wd, available_metrics
            ),
        }


class TrainDataSource(TypedDict):
    data_root: str
    ann_file: str
    img_prefix: str


def _normalize_ann_relpath(data_root: Path, ann: str) -> str:
    """MMDet は data_root からの相対パス（例: annotations/instances_*.json）を想定する。"""
    rel = ann.strip().replace("\\", "/").lstrip("/")
    if (data_root / rel).is_file():
        return rel
    base = Path(rel).name
    under_ann = f"annotations/{base}"
    if (data_root / under_ann).is_file():
        return under_ann
    return rel


def _resolve_unlabeled_sources(
    items: Sequence[Dict[str, str]],
) -> List[TrainDataSource]:
    if not items:
        raise ValueError("Unlabeled データを 1 件以上選択してください。")
    root = unlabeled_root().resolve()
    resolved: List[TrainDataSource] = []
    for i, raw in enumerate(items):
        rel_root = (raw.get("data_root_rel") or "").strip()
        ann = (raw.get("ann_file") or "annotations/instances_unlabeled.json").strip()
        prefix = (raw.get("img_prefix") or "images/").strip()
        if not rel_root:
            raise ValueError(f"unlabeled[{i}]: data_root_rel が必要です。")
        data_root = (ROOT / rel_root).resolve()
        if not _is_under(data_root, root):
            raise ValueError(
                f"unlabeled[{i}]: data_root は data/unlabeled 配下のみ選択できます。"
            )
        ann_n = _normalize_ann_relpath(data_root, ann)
        ann_path = data_root / ann_n
        if not ann_path.is_file():
            raise ValueError(f"unlabeled[{i}]: アノテーションが見つかりません: {ann}")
        resolved.append(
            {
                "data_root": str(data_root),
                "ann_file": ann_n,
                "img_prefix": prefix,
            }
        )
    return resolved


def _resolve_data_sources(
    items: Sequence[Dict[str, str]],
    *,
    label: str,
) -> List[TrainDataSource]:
    if not items:
        raise ValueError(f"{label} を 1 件以上選択してください。")
    resolved: List[TrainDataSource] = []
    for i, raw in enumerate(items):
        rel_root = (raw.get("data_root_rel") or "").strip()
        ann = (raw.get("ann_file") or "").strip()
        prefix = (raw.get("img_prefix") or "").strip()
        if not rel_root or not ann or not prefix:
            raise ValueError(f"{label}[{i}]: data_root_rel, ann_file, img_prefix が必要です。")
        data_root = (ROOT / rel_root).resolve()
        if not _is_under(data_root, EXPORTS_DIR.resolve()):
            raise ValueError(f"{label}[{i}]: data_root は data/exports 配下のみ選択できます。")
        ann_n = _normalize_ann_relpath(data_root, ann)
        ann_path = data_root / ann_n
        if not ann_path.is_file():
            raise ValueError(f"{label}[{i}]: アノテーションが見つかりません: {ann}")
        resolved.append(
            {
                "data_root": str(data_root),
                "ann_file": ann_n,
                "img_prefix": prefix,
            }
        )
    return resolved


def suggest_categories_for_request(
    train_sources: Sequence[Dict[str, str]],
    val_sources: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    train_r = _resolve_data_sources(train_sources, label="train データ") if train_sources else []
    val_r = _resolve_data_sources(val_sources, label="val データ") if val_sources else []
    if not train_r and not val_r:
        return {"categories": [], "names": [], "default_selected": []}
    return suggest_categories(train_r, val_r)


class TrainJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[TrainJob] = None

    def active_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._job is None:
                return None
            return self._job.snapshot()

    def start(
        self,
        *,
        train_sources: Sequence[Dict[str, str]],
        val_sources: Sequence[Dict[str, str]],
        max_epochs: int,
        lr: float,
        batch_size: int,
        config_path: Optional[str] = None,
        pretrained_work_dir: Optional[str] = None,
        work_dir_name: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        unlabeled_sources: Optional[Sequence[Dict[str, str]]] = None,
        max_iters: Optional[int] = None,
        val_interval: Optional[int] = None,
    ) -> TrainJob:
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise RuntimeError("既に学習が実行中です。完了するまで待つか、停止してください。")

        train_resolved = _resolve_data_sources(train_sources, label="train データ")
        val_resolved = _resolve_data_sources(val_sources, label="val データ")

        cfg_path = resolve_mmdet_config_path(config_path)
        cfg_rel = cfg_path.relative_to(MMDET_ROOT.resolve()).as_posix()
        semi = is_semi_config_rel(cfg_rel)

        unlabeled_resolved: List[TrainDataSource] = []
        if semi:
            if not unlabeled_sources:
                raise ValueError(
                    "半教師あり config では Unlabeled データを 1 件以上選択してください。"
                )
            unlabeled_resolved = _resolve_unlabeled_sources(unlabeled_sources)

        if semi and batch_size < 2:
            raise ValueError(
                "半教師あり（Soft-Teacher）では batch_size を 2 以上にしてください。"
                " 推奨は 5（labeled:unlabeled = 1:4）です。"
            )

        if semi:
            resolved_max_iters = int(max_iters or DEFAULT_SEMI_MAX_ITERS)
            resolved_val_interval = _resolve_val_interval(
                val_interval,
                limit=resolved_max_iters,
                default=DEFAULT_SEMI_VAL_INTERVAL,
                label="val_interval (iter)",
            )
        else:
            resolved_val_interval = _resolve_val_interval(
                val_interval,
                limit=max_epochs,
                default=DEFAULT_VAL_INTERVAL_EPOCHS,
                label="val_interval (epoch)",
            )

        pretrained_checkpoint: Optional[str] = None
        if pretrained_work_dir and pretrained_work_dir.strip():
            if semi:
                raise ValueError(
                    "半教師あり（Soft-Teacher）では work_dirs からの事前学習重みは未対応です。"
                    " config の COCO 事前学習 backbone を使用します。"
                )
            wd_src = resolve_work_dir(pretrained_work_dir.strip())
            ckpt_path, _ = _resolve_checkpoint_path(wd_src)
            pretrained_checkpoint = str(ckpt_path)

        WORK_DIR_PARENT.mkdir(parents=True, exist_ok=True)
        work_dir = allocate_work_dir(work_dir_name)
        if not _is_under(work_dir, WORK_DIR_PARENT.resolve()):
            raise ValueError("work_dir の配置が不正です。")

        cat_info = suggest_categories(train_resolved, val_resolved)
        selected_classes = validate_selected_classes(
            classes, allowed_names=cat_info["names"]
        )

        val_evaluator_ann = prepare_val_evaluator_ann_file(
            val_resolved, work_dir, selected_classes
        )

        categories = build_categories_for_class_names(
            Path(train_resolved[0]["data_root"]) / train_resolved[0]["ann_file"],
            selected_classes,
        )
        unlabeled_prepared: List[Dict[str, str]] = []
        if semi:
            unlabeled_prepared = prepare_unlabeled_ann_files(
                unlabeled_resolved, work_dir, categories
            )

        spec = {
            "config_path": str(cfg_path),
            "train_sources": train_resolved,
            "val_sources": val_resolved,
            "classes": selected_classes,
            "val_evaluator_ann_file": val_evaluator_ann,
            "max_epochs": max_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "work_dir": str(work_dir),
            "semi": semi,
            "val_interval": resolved_val_interval,
        }
        if semi:
            spec["unlabeled_sources"] = unlabeled_prepared
            spec["max_iters"] = resolved_max_iters
        if pretrained_checkpoint:
            spec["pretrained_checkpoint"] = pretrained_checkpoint
            spec["pretrained_work_dir"] = pretrained_work_dir.strip()

        spec_path = work_dir / "webui_train_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "webui.mmdet_train_worker",
            str(spec_path),
        ]
        env = os.environ.copy()
        sep = os.pathsep
        extra = sep.join([str(MMDET_ROOT), str(ROOT)])
        env["PYTHONPATH"] = extra + sep + env.get("PYTHONPATH", "")
        env.setdefault("LOCAL_RANK", "0")

        job = TrainJob(job_id=uuid.uuid4().hex[:12], status="running", work_dir=str(work_dir), command=cmd)

        def _reader() -> None:
            try:
                job._proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert job._proc.stdout is not None
                for line in job._proc.stdout:
                    job.append_line(line)
                code = job._proc.wait()
                with job._lock:
                    if job.status == "stopped":
                        return
                    job.status = "completed" if code == 0 else "failed"
                    if code != 0:
                        job.error = f"プロセス終了コード {code}"
            except Exception as exc:
                with job._lock:
                    job.status = "failed"
                    job.error = str(exc)

        t = threading.Thread(target=_reader, daemon=True)
        job._reader_t = t

        with self._lock:
            self._job = job
        t.start()
        return job

    def stop(self) -> bool:
        with self._lock:
            job = self._job
        if job is None or job.status != "running":
            return False
        proc = job._proc
        if proc is None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        with job._lock:
            job.status = "stopped"
        return True


train_manager = TrainJobManager()
