"""PPAL COCO active-learning training defaults (aligned with upstream PPAL)."""

from __future__ import annotations

# al_configs/coco/ppal_retinanet_coco.py → al_train/retinanet_26e.py
DEFAULT_PPAL_CONFIG_REL = "configs/coco_active_learning/al_train/retinanet_26e.py"

DEFAULT_MAX_EPOCHS = 26
DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE = 1
DEFAULT_WORKERS_PER_GPU = 2
DEFAULT_LR_WARMUP_ITERS = 500
DEFAULT_LR_STEP_EPOCH = 20  # lr_config.step in retinanet_26e.py
DEFAULT_EVAL_INTERVAL = 1  # validate every epoch (retinanet_26e sets 999999999 for AL loop)
DEFAULT_LOG_INTERVAL = 10  # TextLoggerHook interval (smaller than PPAL default 50 for Web UI)
