"""Run MMDetection training from a JSON spec (invoked as subprocess by the Web UI)."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mmdet_root() -> Path:
    return _repo_root() / "third_party" / "mmdetection"


def _ensure_paths() -> None:
    root = _repo_root()
    mmdet = _mmdet_root()
    os.environ.setdefault("LOCAL_RANK", "0")
    for p in (str(mmdet), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(str(mmdet))


def _legacy_sources(spec: dict, split: str) -> List[Dict[str, str]]:
    """Backward compat: single data_root + ann + prefix fields."""
    dr = spec.get("data_root")
    if split == "train":
        ann, prefix = spec.get("train_ann"), spec.get("train_img_prefix")
    else:
        ann, prefix = spec.get("val_ann"), spec.get("val_img_prefix")
    if not dr or not ann or not prefix:
        return []
    return [{"data_root": dr, "ann_file": ann, "img_prefix": prefix}]


def _sources_from_spec(spec: dict, split: str) -> List[Dict[str, str]]:
    key = f"{split}_sources"
    rows = spec.get(key)
    if isinstance(rows, list) and rows:
        return rows
    return _legacy_sources(spec, split)


def _coco_subdataset(
    template: dict,
    source: Dict[str, str],
    classes: Optional[List[str]] = None,
) -> dict:
    ds = copy.deepcopy(template)
    ds["data_root"] = source["data_root"]
    ds["ann_file"] = source["ann_file"]
    ds["data_prefix"] = dict(img=source["img_prefix"])
    if classes:
        ds["metainfo"] = dict(classes=tuple(classes))
    return ds


def _build_dataset_cfg(
    template: dict,
    sources: List[Dict[str, str]],
    classes: Optional[List[str]] = None,
) -> dict:
    if len(sources) == 1:
        return _coco_subdataset(template, sources[0], classes)
    return {
        "type": "ConcatDataset",
        "datasets": [_coco_subdataset(template, s, classes) for s in sources],
    }


def _extract_coco_template(dataset_cfg: Any) -> dict:
    """CocoDataset 相当の dict をテンプレートとして取り出す。"""
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset 設定が dict ではありません。")
    ds_type = dataset_cfg.get("type")
    if ds_type == "MultiImageMixDataset":
        return _extract_coco_template(dataset_cfg.get("dataset", {}))
    if ds_type == "ConcatDataset":
        subs = dataset_cfg.get("datasets") or []
        if not subs:
            raise ValueError("ConcatDataset に datasets がありません。")
        return _extract_coco_template(subs[0])
    if "ann_file" in dataset_cfg:
        return copy.deepcopy(dataset_cfg)
    raise ValueError(
        f"データセット上書き未対応の type です: {ds_type!r}。"
        " CocoDataset 系 config を選ぶか、yolox_s_finetune.py を使用してください。"
    )


def _is_semi_config(cfg) -> bool:
    model_type = cfg.model.get("type")
    return model_type in ("SoftTeacher", "SemiBaseDetector")


def _extract_semi_templates(dataset_cfg: dict) -> tuple:
    if dataset_cfg.get("type") != "ConcatDataset":
        raise ValueError(
            "半教師あり config では train_dataloader.dataset が "
            "ConcatDataset([labeled, unlabeled]) である必要があります。"
        )
    subs = dataset_cfg.get("datasets") or []
    if len(subs) < 2:
        raise ValueError("半教師あり ConcatDataset に labeled / unlabeled がありません。")
    return copy.deepcopy(subs[0]), copy.deepcopy(subs[1])


def _apply_model_classes(cfg, classes: List[str]) -> None:
    n = len(classes)
    model_type = cfg.model.get("type")
    if model_type in ("SoftTeacher", "SemiBaseDetector"):
        detector = cfg.model.get("detector")
        if detector is None:
            raise ValueError("半教師あり model に detector がありません。")
        bbox_head = detector.get("roi_head", {}).get("bbox_head")
        if bbox_head is None:
            raise ValueError("detector.roi_head.bbox_head がありません。")
        bbox_head.num_classes = n
        return
    bbox_head = cfg.model.get("bbox_head")
    if bbox_head is not None:
        bbox_head.num_classes = n


def _set_semi_train_datasets(
    cfg,
    labeled_sources: List[Dict[str, str]],
    unlabeled_sources: List[Dict[str, str]],
    classes: Optional[List[str]] = None,
) -> None:
    train_ds = cfg.train_dataloader.dataset
    labeled_tpl, unlabeled_tpl = _extract_semi_templates(train_ds)
    labeled_built = _build_dataset_cfg(labeled_tpl, labeled_sources, classes)
    unlabeled_built = _build_unlabeled_dataset_cfg(unlabeled_tpl, unlabeled_sources)
    train_ds.datasets = [labeled_built, unlabeled_built]


def _build_unlabeled_dataset_cfg(
    template: dict,
    sources: List[Dict[str, str]],
) -> dict:
    if len(sources) == 1:
        return _coco_unlabeled_subdataset(template, sources[0])
    return {
        "type": "ConcatDataset",
        "datasets": [_coco_unlabeled_subdataset(template, s) for s in sources],
    }


def _coco_unlabeled_subdataset(template: dict, source: Dict[str, str]) -> dict:
    ds = copy.deepcopy(template)
    ds["data_root"] = source["data_root"]
    ann = source["ann_file"]
    ds["ann_file"] = ann if Path(ann).is_absolute() else ann
    ds["data_prefix"] = dict(img=source["img_prefix"])
    ds.pop("metainfo", None)
    filter_cfg = ds.get("filter_cfg")
    if isinstance(filter_cfg, dict):
        filter_cfg["filter_empty_gt"] = False
    else:
        ds["filter_cfg"] = dict(filter_empty_gt=False)
    return ds


def _set_train_dataset(
    cfg, sources: List[Dict[str, str]], classes: Optional[List[str]] = None
) -> None:
    train_ds = cfg.train_dataloader.dataset
    template = _extract_coco_template(train_ds)
    built = _build_dataset_cfg(template, sources, classes)
    if _dataset_type(train_ds) == "MultiImageMixDataset":
        train_ds.dataset = built
    else:
        cfg.train_dataloader.dataset = built


def _dataset_type(dataset_cfg: Any) -> Optional[str]:
    if isinstance(dataset_cfg, dict):
        return dataset_cfg.get("type")
    return getattr(dataset_cfg, "type", None)


def _set_eval_dataset(
    cfg, attr: str, sources: List[Dict[str, str]], classes: Optional[List[str]] = None
) -> None:
    loader = cfg.get(attr)
    if loader is None:
        return
    template = _extract_coco_template(loader.dataset)
    loader.dataset = _build_dataset_cfg(template, sources, classes)


def _apply_semi_param_scheduler(cfg, max_iters: int, lr: float) -> None:
    sched = cfg.param_scheduler
    if not sched or len(sched) < 2:
        return
    second = sched[1]
    if not isinstance(second, dict):
        return
    orig_end = int(second.get("end") or max_iters)
    if orig_end <= 0:
        orig_end = max_iters
    second["end"] = max_iters
    milestones = second.get("milestones")
    if isinstance(milestones, list) and milestones:
        second["milestones"] = sorted(
            {
                max(1, min(max_iters - 1, int(max_iters * (m / orig_end))))
                for m in milestones
            }
        )


def _semi_source_ratio(batch_size: int, preferred=(1, 4)) -> List[int]:
    if batch_size < 2:
        raise ValueError(
            "半教師あり学習では batch_size は 2 以上必要です "
            "（各バッチに labeled 1 + unlabeled 1 以上）。"
        )
    if batch_size >= sum(preferred):
        return list(preferred)
    return [1, batch_size - 1]


def _ensure_iter_val_logging(cfg) -> None:
    """IterBasedTrainLoop では val mAP を iter 軸で記録する（既定は epoch=0 固定になる）。"""
    hooks = cfg.get("default_hooks")
    if hooks is None:
        cfg.default_hooks = dict()
        hooks = cfg.default_hooks
    logger_hook = hooks.get("logger")
    if logger_hook is None:
        hooks["logger"] = dict(
            type="LoggerHook", interval=50, log_metric_by_epoch=False
        )
        return
    if isinstance(logger_hook, dict):
        logger_hook["log_metric_by_epoch"] = False
    else:
        logger_hook.log_metric_by_epoch = False


def _sync_semi_train_dataloader(cfg, batch_size: int) -> None:
    """GroupMultiSourceSampler.batch_size を DataLoader と揃える（不一致で空バッチになる）。"""
    loader = cfg.train_dataloader
    loader.batch_size = batch_size
    sampler = loader.get("sampler")
    if sampler is None:
        raise ValueError(
            "半教師あり config の train_dataloader.sampler がありません。"
        )
    sampler_type = sampler.get("type", "")
    if "MultiSourceSampler" not in sampler_type:
        raise ValueError(
            f"半教師あり学習では MultiSourceSampler 系が必要です: {sampler_type!r}"
        )
    preferred = tuple(sampler.get("source_ratio") or [1, 4])
    sampler.batch_size = batch_size
    sampler.source_ratio = _semi_source_ratio(batch_size, preferred=preferred)


def _apply_yolox_param_scheduler(cfg, max_epochs: int, lr: float) -> None:
    sched = cfg.param_scheduler
    if not sched or len(sched) < 3:
        return
    nel = int(cfg.get("num_last_epochs", 5))
    nel = max(1, min(nel, max_epochs - 1))
    cos_end = max_epochs - nel
    warm_end = min(5, max(1, cos_end - 1))
    cos_begin = 5 if cos_end > 6 else max(1, cos_end - 1)
    sched[0]["end"] = warm_end
    sched[1]["begin"] = cos_begin
    sched[1]["end"] = cos_end
    sched[1]["T_max"] = cos_end
    sched[1]["eta_min"] = lr * 0.05
    sched[2]["begin"] = cos_end
    sched[2]["end"] = max_epochs


def _apply_spec(cfg, spec: dict) -> None:
    train_sources = _sources_from_spec(spec, "train")
    val_sources = _sources_from_spec(spec, "val")
    if not train_sources:
        raise ValueError("train_sources が空です。")
    if not val_sources:
        raise ValueError("val_sources が空です。")

    max_epochs = int(spec["max_epochs"])
    lr = float(spec["lr"])
    batch_size = int(spec["batch_size"])
    work_dir = spec["work_dir"]
    classes = spec.get("classes")
    if classes is not None and not isinstance(classes, list):
        raise ValueError("classes は文字列のリストである必要があります。")
    class_list = list(classes) if classes else None
    semi = bool(spec.get("semi")) or _is_semi_config(cfg)

    if semi:
        _sync_semi_train_dataloader(cfg, batch_size)
    else:
        cfg.train_dataloader.batch_size = batch_size
    cfg.val_dataloader.batch_size = min(int(batch_size), 1) if semi else batch_size

    if class_list:
        _apply_model_classes(cfg, class_list)

    if semi:
        unlabeled_sources = spec.get("unlabeled_sources") or []
        if not unlabeled_sources:
            raise ValueError("半教師あり学習には unlabeled_sources が必要です。")
        _set_semi_train_datasets(cfg, train_sources, unlabeled_sources, class_list)
    else:
        _set_train_dataset(cfg, train_sources, class_list)

    _set_eval_dataset(cfg, "val_dataloader", val_sources, class_list)
    _set_eval_dataset(cfg, "test_dataloader", val_sources, class_list)

    ann_val_abs = spec.get("val_evaluator_ann_file")
    if not ann_val_abs:
        s0 = val_sources[0]
        ann_val_abs = str(Path(s0["data_root"]) / s0["ann_file"])
    if cfg.get("val_evaluator") is not None:
        cfg.val_evaluator.ann_file = ann_val_abs
    if cfg.get("test_evaluator") is not None:
        cfg.test_evaluator.ann_file = ann_val_abs

    if semi:
        max_iters = int(spec.get("max_iters") or cfg.train_cfg.max_iters)
        if cfg.get("train_cfg") is not None:
            cfg.train_cfg.max_iters = max_iters
            if spec.get("val_interval") is not None:
                cfg.train_cfg.val_interval = int(spec["val_interval"])
        _apply_semi_param_scheduler(cfg, max_iters, lr)
        _ensure_iter_val_logging(cfg)
    elif cfg.get("train_cfg") is not None:
        if cfg.train_cfg.get("max_epochs") is not None:
            cfg.train_cfg.max_epochs = max_epochs
        if spec.get("val_interval") is not None:
            cfg.train_cfg.val_interval = int(spec["val_interval"])

    if cfg.get("optim_wrapper") is not None and cfg.optim_wrapper.get("optimizer") is not None:
        cfg.optim_wrapper.optimizer.lr = lr

    if not semi:
        _apply_yolox_param_scheduler(cfg, max_epochs, lr)

    asl = cfg.get("auto_scale_lr")
    if asl is not None:
        asl["base_batch_size"] = batch_size

    cfg.work_dir = work_dir

    pretrained = spec.get("pretrained_checkpoint")
    if pretrained:
        init_cfg = cfg.model.get("init_cfg")
        if init_cfg is None and _is_semi_config(cfg):
            detector = cfg.model.get("detector") or {}
            init_cfg = detector.get("init_cfg")
            if init_cfg is None:
                backbone = detector.get("backbone") or {}
                init_cfg = backbone.get("init_cfg")
        if init_cfg is None:
            raise ValueError(
                "pretrained_checkpoint を指定しましたが、config に model.init_cfg がありません。"
            )
        if isinstance(init_cfg, dict):
            init_cfg["checkpoint"] = str(Path(pretrained).resolve())
        else:
            raise ValueError("model.init_cfg の形式が想定外です（dict を想定）。")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m webui.mmdet_train_worker <spec.json>", flush=True)
        return 2
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    _ensure_paths()

    from mmengine.config import Config
    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    from mmdet.utils import setup_cache_size_limit_of_dynamo

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(spec["config_path"])
    cfg.launcher = "none"
    _apply_spec(cfg, spec)

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.train()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc
