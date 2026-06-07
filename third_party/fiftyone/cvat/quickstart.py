import fiftyone as fo
import fiftyone.zoo as foz
import time

dataset = foz.load_zoo_dataset("quickstart")

# 例: 検出タスクをCVATへ
anno_key = "cvat_run_001"
wait_for_completion = True
poll_interval_sec = 10
timeout_sec = 60 * 60

dataset.annotate(
    anno_key,
    label_field="ground_truth",
    backend="cvat",
    label_type="detections",   # tasksに応じて変更
    classes=["person", "car", "dog"],
)

print(f"Submitted annotation job to CVAT: anno_key='{anno_key}'")

if wait_for_completion:
    print("Waiting for CVAT annotation completion...")
    deadline = time.time() + timeout_sec
    last_error = None

    while time.time() < deadline:
        try:
            dataset.load_annotations(anno_key)
            print("Loaded annotations from CVAT into FiftyOne.")
            break
        except Exception as e:
            # CVAT側がまだ作業中の場合は一定間隔で再試行
            last_error = e
            print(f"Not ready yet: {e}")
            time.sleep(poll_interval_sec)
    else:
        raise TimeoutError(
            f"Timed out after {timeout_sec}s waiting for CVAT completion"
        ) from last_error
else:
    print(
        "Run dataset.load_annotations(anno_key) after completing annotation on CVAT."
    )