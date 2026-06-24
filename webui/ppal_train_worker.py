"""Run PPAL RetinaNet training from a JSON spec (subprocess entry for the Web UI)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from webui.ppal_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_LR,
    DEFAULT_LR_STEP_EPOCH,
    DEFAULT_LR_WARMUP_ITERS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PPAL_CONFIG_REL,
    DEFAULT_WORKERS_PER_GPU,
)


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
    work_dir: Path,
    out_name: str,
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
    out = work_dir / out_name
    out.write_text(json.dumps(raw), encoding="utf-8")
    return out


def _prepare_unlabeled_json(labeled_path: Path, work_dir: Path) -> Path:
    labeled = json.loads(labeled_path.read_text(encoding="utf-8"))
    out = work_dir / "unlabeled.json"
    payload = {
        "images": [],
        "annotations": [],
        "categories": labeled.get("categories") or [],
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _resolve_img_prefix(data_root: Path, img_prefix: str) -> str:
    rel = img_prefix.strip().replace("\\", "/").strip("/")
    prefix = str((data_root / rel).resolve())
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _format_cfg_classes(class_names: List[str]) -> str:
    inner = ",".join(repr(name) for name in class_names)
    return f"({inner},)" if len(class_names) == 1 else f"({inner})"


def _lr_step_for_epochs(max_epochs: int) -> int:
    if max_epochs == DEFAULT_MAX_EPOCHS:
        return DEFAULT_LR_STEP_EPOCH
    return max(1, round(max_epochs * DEFAULT_LR_STEP_EPOCH / DEFAULT_MAX_EPOCHS))


def _master_port(work_dir: Path) -> int:
    return 29500 + (hash(str(work_dir.resolve())) % 500)


def _build_cfg_options(
    spec: Dict[str, Any],
    labeled: Path,
    unlabeled: Path,
    val_json: Path,
    train_img_prefix: str,
    val_img_prefix: str,
) -> List[str]:
    classes = spec.get("classes") or []
    num_classes = len(classes)
    if num_classes < 1:
        raise ValueError("classes が空です。")

    max_epochs = int(spec.get("max_epochs", DEFAULT_MAX_EPOCHS))
    lr = float(spec.get("lr", DEFAULT_LR))
    batch_size = int(spec.get("batch_size", DEFAULT_BATCH_SIZE))
    lr_step = _lr_step_for_epochs(max_epochs)
    classes_opt = _format_cfg_classes(classes)

    return [
        f"labeled_data={labeled}",
        f"unlabeled_data={unlabeled}",
        f"data.train.ann_file={labeled}",
        f"data.train.img_prefix={train_img_prefix}",
        f"data.train.classes={classes_opt}",
        f"data.val.ann_file={val_json}",
        f"data.val.img_prefix={val_img_prefix}",
        f"data.val.classes={classes_opt}",
        f"data.test.ann_file={val_json}",
        f"data.test.img_prefix={val_img_prefix}",
        f"data.test.classes={classes_opt}",
        f"model.bbox_head.num_classes={num_classes}",
        f"runner.max_epochs={max_epochs}",
        f"data.samples_per_gpu={batch_size}",
        f"data.workers_per_gpu={DEFAULT_WORKERS_PER_GPU}",
        f"optimizer.lr={lr}",
        f"lr_config.step=[{lr_step}]",
        f"lr_config.warmup_iters={DEFAULT_LR_WARMUP_ITERS}",
        f"checkpoint_config.interval={max_epochs}",
        f"evaluation.interval={DEFAULT_EVAL_INTERVAL}",
        f"evaluation.metric=bbox",
        f"log_config.interval={DEFAULT_LOG_INTERVAL}",
    ]


def _validate_ppal_checkpoint(path: Path) -> None:
    import torch

    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "bbox_head.class_quality" not in state:
        raise ValueError(
            "学習結果に bbox_head.class_quality がありません。"
            " PPAL RetinaQualityEMAHead での学習に失敗した可能性があります。"
        )


def _find_checkpoint(work_dir: Path) -> Path:
    latest = work_dir / "latest.pth"
    if latest.is_file() or latest.is_symlink():
        return latest.resolve()
    candidates = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1].resolve()
    raise FileNotFoundError(f"checkpoint が見つかりません: {work_dir}")


def _load_source(spec: Dict[str, Any], key: str) -> tuple[Path, str, str]:
    data_root = Path(spec[f"{key}_data_root"]).resolve()
    ann_path = data_root / spec[f"{key}_ann_file"]
    if not ann_path.is_file():
        raise FileNotFoundError(f"{key} アノテーションが見つかりません: {ann_path}")
    img_prefix = _resolve_img_prefix(data_root, spec[f"{key}_img_prefix"])
    return data_root, str(ann_path), img_prefix


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

    _, labeled_ann, train_img_prefix = _load_source(spec, "labeled")
    _, val_ann, val_img_prefix = _load_source(spec, "val")
    classes = spec.get("classes")

    labeled = _prepare_coco_json(
        Path(labeled_ann), work_dir, "labeled.json", classes, require_annotations=True
    )
    val_json = _prepare_coco_json(
        Path(val_ann), work_dir, "val.json", classes, require_annotations=True
    )
    unlabeled = _prepare_unlabeled_json(labeled, work_dir)

    config_rel = (spec.get("config_path") or DEFAULT_PPAL_CONFIG_REL).strip()
    config_path = (ppal_root / config_rel).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"PPAL config が見つかりません: {config_path}")

    cfg_options = _build_cfg_options(
        spec, labeled, unlabeled, val_json, train_img_prefix, val_img_prefix
    )

    port = _master_port(work_dir)
    cmd: List[str] = [
        _resolve_ppal_python(),
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=1",
        f"--master_port={port}",
        "tools/train.py",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--launcher",
        "pytorch",
        "--cfg-options",
        *cfg_options,
    ]

    resume_from = spec.get("resume_from")
    if resume_from:
        ckpt = Path(resume_from).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"resume checkpoint が見つかりません: {ckpt}")
        cmd.extend(["--resume-from", str(ckpt)])

    print("PPAL train command:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    ppal = str(ppal_root)
    env["PYTHONPATH"] = ppal + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(ppal_root),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        return proc.returncode

    ckpt_path = _find_checkpoint(work_dir)
    _validate_ppal_checkpoint(ckpt_path)
    print(f"PPAL checkpoint OK: {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
