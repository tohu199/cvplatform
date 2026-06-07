# quickstart_fiftyone.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fiftyone as fo
import fiftyone.zoo as foz

from plugin_config import configure_plugins, make_cvat_panel_spaces

configure_plugins()

# 既存データセットがあれば再利用、なければ作成
dataset_name = "quickstart"

if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
else:
    # 例: COCO-2017 の validation 200枚だけ取得
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=200,
        dataset_name=dataset_name,
        shuffle=True,
    )

# アプリ起動（CVAT k-center パネル付き）
session = fo.launch_app(
    dataset,
    port=5151,
    address="0.0.0.0",
    spaces=make_cvat_panel_spaces(),
)

print("FiftyOne app started: http://localhost:5151")
print("CVAT k-center panel opens beside Samples. If missing, use + or operator browser.")
session.wait()  # Ctrl+C まで維持
