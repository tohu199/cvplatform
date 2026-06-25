"""Build PPAL runtime config (ConcatDataset) and launch distributed training.

Invoked with the PPAL venv Python after ``ppal_train_worker`` prepares COCO JSON files.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _ensure_ppal_paths() -> None:
    root = _repo_root()
    ppal = _ppal_root()
    os.environ.setdefault("LOCAL_RANK", "0")
    for p in (str(ppal), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(str(ppal))


def _lr_step_for_epochs(max_epochs: int) -> int:
    if max_epochs == DEFAULT_MAX_EPOCHS:
        return DEFAULT_LR_STEP_EPOCH
    return max(1, round(max_epochs * DEFAULT_LR_STEP_EPOCH / DEFAULT_MAX_EPOCHS))


def _master_port(work_dir: Path) -> int:
    return 29500 + (hash(str(work_dir.resolve())) % 500)


def _classes_tuple(class_names: Sequence[str]) -> tuple:
    return tuple(class_names)


def _extract_coco_template(dataset_cfg: Any) -> dict:
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset 設定が dict ではありません。")
    ds_type = dataset_cfg.get("type")
    if ds_type == "ConcatDataset":
        subs = dataset_cfg.get("datasets") or []
        if not subs:
            raise ValueError("ConcatDataset に datasets がありません。")
        return _extract_coco_template(subs[0])
    if ds_type == "MultiImageMixDataset":
        return _extract_coco_template(dataset_cfg.get("dataset", {}))
    if "ann_file" in dataset_cfg or ds_type in (None, "CocoDataset"):
        return copy.deepcopy(dataset_cfg)
    raise ValueError(
        f"データセット上書き未対応の type です: {ds_type!r}。"
        " CocoDataset 系 config を使用してください。"
    )


def _ppal_coco_subdataset(
    template: dict,
    *,
    ann_file: str,
    img_prefix: str,
    classes: Sequence[str],
) -> dict:
    ds = copy.deepcopy(template)
    ds["type"] = ds.get("type") or "CocoDataset"
    ds["ann_file"] = ann_file
    ds["img_prefix"] = img_prefix
    ds["classes"] = _classes_tuple(classes)
    return ds


def _build_dataset_cfg(
    template: dict,
    prepared: Sequence[Dict[str, str]],
    classes: Sequence[str],
) -> dict:
    if len(prepared) == 1:
        row = prepared[0]
        return _ppal_coco_subdataset(
            template,
            ann_file=row["ann_file"],
            img_prefix=row["img_prefix"],
            classes=classes,
        )
    return {
        "type": "ConcatDataset",
        "datasets": [
            _ppal_coco_subdataset(
                template,
                ann_file=row["ann_file"],
                img_prefix=row["img_prefix"],
                classes=classes,
            )
            for row in prepared
        ],
        "separate_eval": True,
    }


def _set_train_dataset(cfg, prepared: Sequence[Dict[str, str]], classes: Sequence[str]) -> None:
    template = _extract_coco_template(cfg.data.train)
    cfg.data.train = _build_dataset_cfg(template, prepared, classes)


def _set_eval_dataset(
    cfg,
    split: str,
    prepared: Sequence[Dict[str, str]],
    classes: Sequence[str],
) -> None:
    template = _extract_coco_template(cfg.data[split])
    cfg.data[split] = _build_dataset_cfg(template, prepared, classes)


def _apply_spec(cfg, spec: Dict[str, Any]) -> None:
    classes = spec.get("classes") or []
    if not classes:
        raise ValueError("classes が空です。")

    labeled_prepared = spec.get("labeled_prepared") or []
    val_prepared = spec.get("val_prepared") or []
    if not labeled_prepared:
        raise ValueError("labeled_prepared が空です。")
    if not val_prepared:
        raise ValueError("val_prepared が空です。")

    max_epochs = int(spec.get("max_epochs", DEFAULT_MAX_EPOCHS))
    lr = float(spec.get("lr", DEFAULT_LR))
    batch_size = int(spec.get("batch_size", DEFAULT_BATCH_SIZE))
    lr_step = _lr_step_for_epochs(max_epochs)
    num_classes = len(classes)

    cfg.labeled_data = spec.get("labeled_data") or labeled_prepared[0]["ann_file"]
    cfg.unlabeled_data = spec.get("unlabeled_data") or ""

    _set_train_dataset(cfg, labeled_prepared, classes)
    _set_eval_dataset(cfg, "val", val_prepared, classes)
    _set_eval_dataset(cfg, "test", val_prepared, classes)

    cfg.model.bbox_head.num_classes = num_classes
    cfg.runner.max_epochs = max_epochs
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = DEFAULT_WORKERS_PER_GPU
    cfg.optimizer.lr = lr
    cfg.lr_config.step = [lr_step]
    cfg.lr_config.warmup_iters = DEFAULT_LR_WARMUP_ITERS
    cfg.checkpoint_config.interval = max_epochs
    cfg.evaluation.interval = DEFAULT_EVAL_INTERVAL
    cfg.evaluation.metric = "bbox"
    cfg.log_config.interval = DEFAULT_LOG_INTERVAL
    cfg.work_dir = spec["work_dir"]


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


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m webui.ppal_train_launch <launch_spec.json>", flush=True)
        return 2

    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    ppal_root = _ppal_root()
    work_dir = Path(spec["work_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    _ensure_ppal_paths()

    from mmcv import Config

    config_rel = (spec.get("config_path") or DEFAULT_PPAL_CONFIG_REL).strip()
    config_path = (ppal_root / config_rel).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"PPAL config が見つかりません: {config_path}")

    cfg = Config.fromfile(str(config_path))
    _apply_spec(cfg, spec)

    runtime_cfg = work_dir / "webui_ppal_runtime.py"
    cfg.dump(str(runtime_cfg))
    print(f"PPAL runtime config: {runtime_cfg}", flush=True)

    port = _master_port(work_dir)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=1",
        f"--master_port={port}",
        "tools/train.py",
        str(runtime_cfg),
        "--work-dir",
        str(work_dir),
        "--launcher",
        "pytorch",
    ]

    resume_from = spec.get("resume_from")
    if resume_from:
        ckpt = Path(resume_from).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"resume checkpoint が見つかりません: {ckpt}")
        cmd.extend(["--resume-from", str(ckpt)])

    print("PPAL train command:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ppal_root), check=False)
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
