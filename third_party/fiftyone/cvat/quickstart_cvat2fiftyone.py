import fiftyone as fo
import fiftyone.zoo as foz

DATASET_NAME = "quickstart"
ANNO_KEY = "cvat_run_001"

if DATASET_NAME in fo.list_datasets():
    dataset = fo.load_dataset(DATASET_NAME)
elif DATASET_NAME == "quickstart":
    print("Dataset 'quickstart' was not found. Creating it from FiftyOne Zoo...")
    dataset = foz.load_zoo_dataset("quickstart")
else:
    raise ValueError(
        f"Dataset '{DATASET_NAME}' not found. "
        "Create/import the dataset first."
    )

annotation_runs = dataset.list_annotation_runs()
if ANNO_KEY not in annotation_runs:
    raise ValueError(
        f"Dataset '{DATASET_NAME}' has no annotation run key '{ANNO_KEY}'. "
        f"available_keys={annotation_runs}"
    )

dataset.load_annotations(ANNO_KEY)

print(f"Loaded CVAT annotations: dataset='{DATASET_NAME}', anno_key='{ANNO_KEY}'")
print(dataset.head())

"""
python third_party/fiftyone/quickstart_cvat2fiftyone.py
"""