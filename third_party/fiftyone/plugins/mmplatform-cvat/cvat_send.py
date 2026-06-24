"""CVAT upload helpers for the mmplatform FiftyOne plugin."""

from __future__ import annotations

import os
from typing import List, Sequence

import fiftyone as fo


def default_anno_key() -> str:
    import time

    return f"mmplatform_cvat_{int(time.time())}"


def cvat_env_summary() -> str:
    url = (os.environ.get("FIFTYONE_CVAT_URL") or "").strip()
    user = (os.environ.get("FIFTYONE_CVAT_USERNAME") or "").strip()
    if not url:
        return "FIFTYONE_CVAT_URL 未設定（既定: https://app.cvat.ai）"
    if not user:
        return f"CVAT URL={url}（ユーザー名未設定）"
    return f"CVAT URL={url}, user={user}"


def validate_cvat_env() -> None:
    url = (os.environ.get("FIFTYONE_CVAT_URL") or "").strip()
    user = (os.environ.get("FIFTYONE_CVAT_USERNAME") or "").strip()
    password = (os.environ.get("FIFTYONE_CVAT_PASSWORD") or "").strip()
    if not url:
        raise ValueError(
            "FIFTYONE_CVAT_URL が未設定です。"
            " ローカル CVAT を使う場合は例: export FIFTYONE_CVAT_URL=http://localhost:8080"
            "（未設定時 FiftyOne は https://app.cvat.ai に接続します）"
        )
    if not user or not password:
        raise ValueError(
            "FIFTYONE_CVAT_USERNAME / FIFTYONE_CVAT_PASSWORD が未設定です。"
            " third_party/fiftyone/personal_info.sh を source するか、"
            " launch_app.py を起動する前に export してください。"
        )


def _clear_annotation_run(dataset: fo.Dataset, anno_key: str) -> bool:
    if not dataset.has_annotation_run(anno_key):
        return False
    dataset.delete_annotation_run(anno_key)
    return True


def format_cvat_send_error(exc: BaseException) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    url = (os.environ.get("FIFTYONE_CVAT_URL") or "").strip() or "https://app.cvat.ai"

    if msg == "EOF" or " EOF" in msg or msg.endswith("EOF"):
        return (
            "CVAT との接続が途中で切れました (EOF)。"
            f" CVAT が起動しているか、URL ({url}) が FiftyOne から到達できるか確認してください。"
            " WSL2 では docker compose の CVAT が localhost:8080 で応答するか"
            " `curl http://localhost:8080/api/server/about` で確認できます。"
        )

    if "already exists" in msg:
        return (
            f"{msg} — 前回の送信が途中で失敗し、同じ Annotation key が残っています。"
            " もう一度「CVAT に送信」を押すと古い run を削除して再試行します"
            "（Annotation key は自動で更新されます）。"
        )

    if "Connection refused" in msg or "Failed to establish a new connection" in msg:
        return (
            f"CVAT に接続できません ({msg})。"
            f" URL={url} が正しいか、CVAT コンテナが起動しているか確認してください。"
        )

    return msg


def send_samples_to_cvat(
    dataset: fo.Dataset,
    sample_ids: Sequence[str],
    *,
    anno_key: str,
    label_field: str,
    classes: List[str],
    launch_editor: bool = True,
) -> str:
    """Upload selected samples to CVAT; clear stale runs and rotate key on failure."""
    validate_cvat_env()
    anno_key = anno_key.strip()
    if not anno_key:
        raise ValueError("Annotation key が空です。")

    cleared = _clear_annotation_run(dataset, anno_key)
    view = dataset.select(list(sample_ids))

    try:
        view.annotate(
            anno_key,
            backend="cvat",
            label_field=label_field,
            label_type="detections",
            classes=classes,
            launch_editor=launch_editor,
        )
    except Exception as exc:
        if dataset.has_annotation_run(anno_key):
            try:
                dataset.delete_annotation_run(anno_key)
            except Exception:
                pass
        raise RuntimeError(format_cvat_send_error(exc)) from exc

    if cleared:
        # Stale run from a prior failed attempt was removed before this send.
        pass

    return anno_key
