"""PPAL RetinaNet training job lifecycle for the Web UI."""

from __future__ import annotations

import json
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
from typing import Any, Deque, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent

from webui.coco_categories import suggest_categories, validate_selected_classes
from webui.ppal_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PPAL_CONFIG_REL,
)
from webui.ppal_metrics import DEFAULT_PPAL_CHART_METRICS, find_ppal_log_json, parse_ppal_log_json
from webui.train_manager import (
    EXPORTS_DIR,
    _is_under,
    _normalize_ann_relpath,
    _resolve_data_sources,
)

PPAL_ROOT = ROOT / "third_party" / "PPAL"
PPAL_WORK_DIRS = PPAL_ROOT / "work_dirs"

WEB_PPAL_PREFIX = "web_ppal_"
WORK_DIR_NAME_RE = re.compile(r"^web_ppal_[A-Za-z0-9_-]+$")

PPAL_EPOCH_LOSS_RE = re.compile(
    r"Epoch\s*\[(\d+)\]\[(\d+)/(\d+)\].*?\bloss:\s+([\d.]+(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
PPAL_ERROR_LINE_RE = re.compile(
    r"(Error|Exception|Traceback|AssertionError|RuntimeError|FileNotFoundError)",
    re.IGNORECASE,
)


def export_stamp() -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M")


def default_work_dir_basename() -> str:
    return f"{WEB_PPAL_PREFIX}{export_stamp()}_{uuid.uuid4().hex[:8]}"


def normalize_work_dir_basename(name: Optional[str]) -> str:
    if not name or not name.strip():
        return default_work_dir_basename()
    n = name.strip()
    if n in (".", "..") or "/" in n or "\\" in n:
        raise ValueError("work_dir 名にパス区切りは使えません。")
    full = n if n.startswith(WEB_PPAL_PREFIX) else f"{WEB_PPAL_PREFIX}{n}"
    if len(full) > 120:
        raise ValueError("work_dir 名は 120 文字以内にしてください。")
    if not WORK_DIR_NAME_RE.match(full):
        raise ValueError(
            "work_dir 名は web_ppal_ の後に英数字と _ - のみ使用できます"
            "（例: web_ppal_task5_round1）。"
        )
    return full


def allocate_work_dir(basename: Optional[str] = None) -> Path:
    name = normalize_work_dir_basename(basename)
    wd = (PPAL_WORK_DIRS / name).resolve()
    if not _is_under(wd, PPAL_WORK_DIRS.resolve()):
        raise ValueError("work_dir の配置が不正です。")
    if wd.exists():
        raise ValueError(f"work_dir が既に存在します: {name}")
    wd.mkdir(parents=True, exist_ok=False)
    return wd


def resolve_ppal_work_dir(name: str) -> Path:
    n = name.strip()
    if not n or n in (".", "..") or "/" in n or "\\" in n:
        raise ValueError("work_dir 名が不正です。")
    wd = (PPAL_WORK_DIRS / n).resolve()
    if not _is_under(wd, PPAL_WORK_DIRS.resolve()):
        raise ValueError("work_dir は third_party/PPAL/work_dirs 配下のみ指定できます。")
    if not wd.is_dir():
        raise ValueError(f"work_dir が見つかりません: {n}")
    return wd


def _resolve_ppal_checkpoint(work_dir: Path) -> Path:
    latest = work_dir / "latest.pth"
    if latest.is_file() or latest.is_symlink():
        return latest.resolve()
    candidates = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1].resolve()
    raise ValueError(f"checkpoint が見つかりません: {work_dir}")


def _has_class_quality(checkpoint: Path) -> bool:
    try:
        import torch

        ckpt = torch.load(str(checkpoint), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        return "bbox_head.class_quality" in state
    except OSError:
        return False


def list_ppal_pretrained_work_dirs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not PPAL_WORK_DIRS.is_dir():
        return rows
    for p in sorted(PPAL_WORK_DIRS.iterdir(), key=lambda x: x.name, reverse=True):
        if not p.is_dir() or not p.name.startswith(WEB_PPAL_PREFIX):
            continue
        if not WORK_DIR_NAME_RE.match(p.name):
            continue
        try:
            ckpt = _resolve_ppal_checkpoint(p)
        except ValueError:
            continue
        rows.append(
            {
                "id": p.name,
                "checkpoint": ckpt.name,
                "checkpoint_path": str(ckpt),
                "ppal_ready": _has_class_quality(ckpt),
            }
        )
    return rows


def ppal_train_defaults() -> Dict[str, Any]:
    return {
        "config_path": DEFAULT_PPAL_CONFIG_REL,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "lr": DEFAULT_LR,
        "batch_size": DEFAULT_BATCH_SIZE,
    }


def resolve_export_source(raw: Dict[str, str], *, label: str) -> Dict[str, str]:
    rel_root = (raw.get("data_root_rel") or "").strip()
    ann = (raw.get("ann_file") or "").strip()
    prefix = (raw.get("img_prefix") or "").strip()
    if not rel_root or not ann or not prefix:
        raise ValueError(f"{label}: data_root_rel, ann_file, img_prefix が必要です。")
    data_root = (ROOT / rel_root).resolve()
    if not _is_under(data_root, EXPORTS_DIR.resolve()):
        raise ValueError(f"{label} は data/exports 配下のみ選択できます。")
    ann_n = _normalize_ann_relpath(data_root, ann)
    ann_path = data_root / ann_n
    if not ann_path.is_file():
        raise ValueError(f"{label}: アノテーションが見つかりません: {ann}")
    return {
        "data_root": str(data_root),
        "ann_file": ann_n,
        "img_prefix": prefix,
    }


def resolve_labeled_sources(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return _resolve_data_sources(items, label="labeled データ")


def resolve_val_sources(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return _resolve_data_sources(items, label="val データ")


def resolve_labeled_source(raw: Dict[str, str]) -> Dict[str, str]:
    return resolve_export_source(raw, label="labeled データ")


def resolve_val_source(raw: Dict[str, str]) -> Dict[str, str]:
    return resolve_export_source(raw, label="val データ")


def suggest_categories_for_ppal(
    labeled_sources: Sequence[Dict[str, str]],
    val_sources: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    labeled = (
        resolve_labeled_sources(labeled_sources) if labeled_sources else []
    )
    val = resolve_val_sources(val_sources) if val_sources else []
    if not labeled and not val:
        return {"categories": [], "names": [], "default_selected": []}
    return suggest_categories(labeled, val)


def suggest_categories_for_labeled(labeled_source: Dict[str, str]) -> Dict[str, Any]:
    """Backward-compatible helper when only labeled is known."""
    resolved = resolve_labeled_source(labeled_source)
    return suggest_categories([resolved], [])


def _summarize_failure(lines: Sequence[str], code: int) -> str:
    for line in reversed(lines[-80:]):
        stripped = line.strip()
        if not stripped:
            continue
        if PPAL_ERROR_LINE_RE.search(stripped):
            return stripped
        if "does not matches" in stripped or "AssertionError" in stripped:
            return stripped
    tail = [ln.strip() for ln in lines[-3:] if ln.strip()]
    if tail:
        return f"プロセス終了コード {code}: " + " | ".join(tail)
    return f"プロセス終了コード {code}"


def _parse_ppal_loss_line(line: str) -> Optional[Dict[str, Any]]:
    m = PPAL_EPOCH_LOSS_RE.search(line)
    if not m:
        return None
    try:
        loss = float(m.group(4))
    except ValueError:
        return None
    return {
        "epoch": int(m.group(1)),
        "iter": int(m.group(2)),
        "iter_total": int(m.group(3)),
        "loss": loss,
    }


@dataclass
class PpalTrainJob:
    job_id: str
    status: str
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
            parsed = _parse_ppal_loss_line(text)
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
        log_json_path: Optional[str] = None
        if wd:
            log_path = find_ppal_log_json(Path(wd))
            if log_path is not None and log_path.is_file():
                metric_series, available_metrics = parse_ppal_log_json(log_path)
                log_json_path = str(log_path)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "work_dir": wd,
            "command": self.command,
            "error": self.error,
            "lines_tail": lst,
            "loss_points": lp,
            "metric_series": metric_series,
            "available_metrics": available_metrics,
            "log_json_path": log_json_path,
            "default_metrics": list(DEFAULT_PPAL_CHART_METRICS),
        }


class PpalTrainJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[PpalTrainJob] = None

    def active_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._job is None:
                return None
            return self._job.snapshot()

    def start(
        self,
        *,
        labeled_sources: Sequence[Dict[str, str]],
        val_sources: Sequence[Dict[str, str]],
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pretrained_work_dir: Optional[str] = None,
        work_dir_name: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> PpalTrainJob:
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise RuntimeError("既に PPAL 学習が実行中です。完了するまで待つか、停止してください。")

        if not (PPAL_ROOT / ".venv" / "bin" / "python").is_file():
            raise ValueError(
                "PPAL 仮想環境が見つかりません。"
                " third_party/PPAL/.venv を作成してください（SETUP.md 参照）。"
            )

        labeled_resolved = resolve_labeled_sources(labeled_sources)
        val_resolved = resolve_val_sources(val_sources)
        cat_info = suggest_categories(labeled_resolved, val_resolved)
        selected_classes = validate_selected_classes(
            classes, allowed_names=cat_info["names"]
        )

        PPAL_WORK_DIRS.mkdir(parents=True, exist_ok=True)
        work_dir = allocate_work_dir(work_dir_name)

        resume_from: Optional[str] = None
        if pretrained_work_dir and pretrained_work_dir.strip():
            wd_src = resolve_ppal_work_dir(pretrained_work_dir.strip())
            ckpt = _resolve_ppal_checkpoint(wd_src)
            resume_from = str(ckpt)

        spec: Dict[str, Any] = {
            "config_path": DEFAULT_PPAL_CONFIG_REL,
            "labeled_sources": labeled_resolved,
            "val_sources": val_resolved,
            "classes": selected_classes,
            "max_epochs": max_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "work_dir": str(work_dir),
        }
        if resume_from:
            spec["resume_from"] = resume_from
            spec["pretrained_work_dir"] = pretrained_work_dir.strip()

        spec_path = work_dir / "webui_ppal_train_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "webui.ppal_train_worker",
            str(spec_path),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        job = PpalTrainJob(
            job_id=uuid.uuid4().hex[:12],
            status="running",
            work_dir=str(work_dir),
            command=cmd,
        )

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
                        with job._lock:
                            err_lines = list(job.lines)
                        job.error = _summarize_failure(err_lines, code)
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


ppal_train_manager = PpalTrainJobManager()
