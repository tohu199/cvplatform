"""k-center greedy ordering helpers for the mmplatform FiftyOne plugin."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import fiftyone as fo
import numpy as np
from sklearn.metrics import pairwise_distances

RANK_FIELD = "kcenter_rank"
DEFAULT_SELECT = 10
MIN_SELECT = 1

EMBEDDING_MODEL_OPTIONS: Tuple[Dict[str, str], ...] = (
    {
        "name": "dinov2-vits14-torch",
        "label": "DINOv2 ViT-S/14",
        "description": "DINOv2 小型。Python 3.9+ が必要",
    },
    {
        "name": "dinov2-vitb14-torch",
        "label": "DINOv2 ViT-B/14",
        "description": "DINOv2 中型。Python 3.9+ が必要",
    },
    {
        "name": "dino-vits16-torch",
        "label": "DINO v1 ViT-S/16",
        "description": "DINO v1 小型 ViT（torch hub）",
    },
    {
        "name": "dino-vitb16-torch",
        "label": "DINO v1 ViT-B/16",
        "description": "DINO v1 中型 ViT（torch hub）",
    },
    {
        "name": "mobilenet-v2-imagenet-torch",
        "label": "MobileNetV2",
        "description": "軽量 CNN。Python 3.8 でも利用可",
    },
    {
        "name": "resnet18-imagenet-torch",
        "label": "ResNet18",
        "description": "CNN。Python 3.8 でも利用可",
    },
)

# Python 3.8 では DINOv2 が型ヒント互換性で失敗するため、デフォルトは MobileNet。
DEFAULT_EMBEDDING_MODEL = (
    "mobilenet-v2-imagenet-torch"
    if sys.version_info < (3, 9)
    else "dinov2-vits14-torch"
)

_DINO_BATCH_SIZE = 4
_CNN_BATCH_SIZE = 8
_model_cache: Dict[str, object] = {}
_DINO_V1_MANIFEST = Path(__file__).resolve().parent / "models" / "dino-v1-manifest.json"
_manifest_registered = False


def _ensure_dino_v1_manifest() -> None:
    global _manifest_registered
    if _manifest_registered or not _DINO_V1_MANIFEST.is_file():
        return

    manifest = str(_DINO_V1_MANIFEST.resolve())
    existing = list(fo.config.model_zoo_manifest_paths or [])
    if manifest not in existing:
        fo.config.model_zoo_manifest_paths = existing + [manifest]
    _manifest_registered = True


@dataclass
class KCenterOrderResult:
    ordered_ids: List[str]
    model_name: str


ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]


@dataclass
class KCenterProgress:
    """Reports k-center pipeline progress and accumulates a text log."""

    on_progress: Optional[ProgressCallback] = None
    on_log: Optional[LogCallback] = None
    _lines: List[str] = field(default_factory=list)
    _last_reported_pct: int = field(default=-1, init=False, repr=False)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self._lines.append(line)
        if self.on_log is not None:
            self.on_log(self.log_text)

    def set(self, fraction: float, label: str, *, force: bool = False) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        pct = int(fraction * 100)
        if not force and pct == self._last_reported_pct:
            return
        self._last_reported_pct = pct
        if self.on_progress is not None:
            self.on_progress(fraction, label)

    @property
    def log_text(self) -> str:
        return "\n".join(self._lines)


def list_embedding_model_names() -> List[str]:
    return [option["name"] for option in EMBEDDING_MODEL_OPTIONS]


def normalize_embedding_model_name(model_name: Optional[str]) -> str:
    name = (model_name or "").strip() or DEFAULT_EMBEDDING_MODEL
    if name not in list_embedding_model_names():
        allowed = ", ".join(list_embedding_model_names())
        raise ValueError(f"未知の embedding モデル: {name}（選択肢: {allowed}）")
    return name


def kcenter_greedy_order(
    sample_ids: Sequence[str],
    embeddings: np.ndarray,
    progress: Optional[KCenterProgress] = None,
) -> List[str]:
    """Return sample IDs ordered by k-center greedy selection."""
    ids = list(sample_ids)
    n = len(ids)
    if n <= 1:
        return ids

    if progress is not None:
        progress.log(f"k-center 並べ替え開始（{n} 枚）")

    emb = np.asarray(embeddings, dtype=float)
    mean = emb.mean(axis=0, keepdims=True)
    first_idx = int(np.argmax(pairwise_distances(emb, mean).ravel()))

    selected_indices = [first_idx]
    min_dists = pairwise_distances(emb, emb[first_idx : first_idx + 1]).ravel()

    for step in range(n - 1):
        min_dists[selected_indices] = -1.0
        next_idx = int(np.argmax(min_dists))
        selected_indices.append(next_idx)
        new_dists = pairwise_distances(emb, emb[next_idx : next_idx + 1]).ravel()
        min_dists = np.minimum(min_dists, new_dists)
        if progress is not None:
            frac = 0.85 + 0.10 * (step + 1) / max(n - 1, 1)
            progress.set(
                frac,
                f"k-center 並べ替え… {step + 1}/{n - 1}",
            )

    if progress is not None:
        progress.log(f"k-center 並べ替え完了（{n} 枚）")

    return [ids[i] for i in selected_indices]


def ensure_rank_field(dataset: fo.Dataset) -> None:
    if RANK_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(RANK_FIELD, fo.IntField)


def set_kcenter_ranks(dataset: fo.Dataset, ordered_ids: Sequence[str]) -> None:
    ensure_rank_field(dataset)
    rank_map = {sample_id: rank for rank, sample_id in enumerate(ordered_ids)}
    dataset.set_values(RANK_FIELD, rank_map, key_field="id")


def _batch_size_for_model(model_name: str) -> int:
    if model_name.startswith(("dinov2-", "dino-")):
        return _DINO_BATCH_SIZE
    return _CNN_BATCH_SIZE


def _format_model_load_error(model_name: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if (
        model_name.startswith("dinov2-")
        and "unsupported operand type(s) for |" in detail
    ):
        py = ".".join(map(str, sys.version_info[:3]))
        return (
            f"embedding モデル '{model_name}' を読み込めませんでした。\n"
            f"DINOv2 は Python 3.9 以上が必要です（現在: Python {py}）。\n"
            "DINO v1 / MobileNet / ResNet を選ぶか、Python 3.9+ 環境で実行してください。\n"
            f"詳細: {detail}"
        )
    return f"embedding モデル '{model_name}' を読み込めませんでした: {detail}"


def _load_embedding_model(model_name: str) -> object:
    model_name = normalize_embedding_model_name(model_name)
    if model_name in _model_cache:
        return _model_cache[model_name]

    _ensure_dino_v1_manifest()
    import fiftyone.zoo as foz

    try:
        model = foz.load_zoo_model(model_name)
    except Exception as exc:
        raise RuntimeError(_format_model_load_error(model_name, exc)) from exc

    _model_cache[model_name] = model
    return model


def _embedding_progress_callback(
    progress: KCenterProgress,
) -> Callable[[object], None]:
    def callback(pb: object) -> None:
        pb_progress = float(getattr(pb, "progress", 0.0))
        pct = int(pb_progress * 100)
        progress.set(
            0.15 + 0.70 * pb_progress,
            f"embedding 計算中… {pct}%",
        )
        if getattr(pb, "complete", False):
            progress.log("embedding 計算完了")

    return callback


def _compute_embeddings(
    view: fo.core.view.DatasetView,
    model_name: str,
    progress: Optional[KCenterProgress] = None,
) -> Tuple[List[str], np.ndarray]:
    model_name = normalize_embedding_model_name(model_name)
    sample_ids = list(view.values("id"))
    if not sample_ids:
        return [], np.empty((0, 0))

    if progress is not None:
        progress.set(0.05, f"モデル読み込み中… ({model_name})")
        progress.log(f"embedding モデル: {model_name}")

    model = _load_embedding_model(model_name)

    if progress is not None:
        batch_size = _batch_size_for_model(model_name)
        progress.set(0.15, f"embedding 計算開始（{len(sample_ids)} 枚, batch={batch_size}）")
        progress.log(
            f"embedding 計算開始: {len(sample_ids)} 枚, batch_size={batch_size}"
        )
        embed_progress = _embedding_progress_callback(progress)
    else:
        embed_progress = False

    embeddings = view.compute_embeddings(
        model,
        batch_size=_batch_size_for_model(model_name),
        num_workers=0,
        skip_failures=False,
        progress=embed_progress,
    )

    if isinstance(embeddings, list):
        if any(item is None for item in embeddings):
            raise RuntimeError(
                "一部サンプルの embedding 計算に失敗しました。"
                "画像パスや形式を確認してください。"
            )
        embeddings = np.asarray(embeddings, dtype=float)

    if len(embeddings) != len(sample_ids):
        raise RuntimeError(
            "embedding 数とサンプル数が一致しません "
            f"({len(embeddings)} != {len(sample_ids)})"
        )

    return sample_ids, np.asarray(embeddings, dtype=float)


def compute_kcenter_order(
    _dataset: fo.Dataset,
    view: fo.core.view.DatasetView,
    model_name: str,
    progress: Optional[KCenterProgress] = None,
) -> KCenterOrderResult:
    """Compute k-center greedy order for samples in *view*."""
    model_name = normalize_embedding_model_name(model_name)
    if progress is not None:
        progress.set(0.0, "k-center 処理を開始…", force=True)
        progress.log(f"対象ビュー: {view.count()} 枚")

    sample_ids, embeddings = _compute_embeddings(view, model_name, progress=progress)
    ordered_ids = kcenter_greedy_order(sample_ids, embeddings, progress=progress)

    if progress is not None:
        progress.set(0.97, "順位を保存中…")
        progress.log("kcenter_rank フィールドへ順位を保存")

    return KCenterOrderResult(ordered_ids=ordered_ids, model_name=model_name)


def selection_bounds_message(count: int) -> str:
    return f"選択: {count} 枚（{MIN_SELECT} 枚以上で CVAT 送信可）"


def validate_selection_count(count: int) -> Tuple[bool, str]:
    if count < MIN_SELECT:
        return False, f"少なくとも {MIN_SELECT} 枚選択してください（現在 {count} 枚）"
    return True, selection_bounds_message(count)
