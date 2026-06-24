"""Parse PPAL (mmdet 2.x) TextLoggerHook *.log.json for Web UI charts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PPAL_LOG_JSON_RE = re.compile(r"^\d{8}_\d{6}\.log\.json$")

DEFAULT_PPAL_CHART_METRICS = ["loss", "memory", "coco/bbox_mAP"]

_SKIP_KEYS = frozenset({"mode", "epoch", "iter"})


def _metric_sort_key(name: str) -> tuple:
    if name == "loss":
        return (0, name)
    if name.startswith("loss_"):
        return (1, name)
    if name == "memory":
        return (2, name)
    if name.startswith("coco/bbox_mAP") or name.startswith("bbox_mAP"):
        return (3, name)
    return (9, name)


def _normalize_metric_key(key: str) -> str:
    if key == "bbox_mAP" or key.startswith("bbox_mAP"):
        return "coco/" + key if not key.startswith("coco/") else key
    return key


def find_ppal_log_json(work_dir: Path) -> Optional[Path]:
    if not work_dir.is_dir():
        return None
    candidates = [
        p
        for p in work_dir.glob("*.log.json")
        if PPAL_LOG_JSON_RE.match(p.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_ppal_log_json(path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    series: Dict[str, List[Dict[str, Any]]] = {}
    keys_seen: set[str] = set()
    step = 0

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "epoch" not in obj:
            continue

        numeric_items = []
        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            fv = float(v)
            if not math.isfinite(fv):
                continue
            numeric_items.append((k, fv))

        if not numeric_items:
            continue

        step += 1
        step_f = float(step)
        for k, fv in numeric_items:
            mk = _normalize_metric_key(k)
            keys_seen.add(mk)
            series.setdefault(mk, []).append({"step": step_f, "value": fv})

    available = sorted(keys_seen, key=_metric_sort_key)
    return series, available
