import argparse
import os

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.cvat as fouc


def main():
    parser = argparse.ArgumentParser(
        description="Import existing CVAT annotations into a FiftyOne dataset"
    )
    parser.add_argument("--dataset", default="quickstart", help="FiftyOne dataset name")
    parser.add_argument(
        "--task-id",
        action="append",
        type=int,
        dest="task_ids",
        default=None,
        help="CVAT task ID (can be specified multiple times)",
    )
    parser.add_argument("--project-name", default=None, help="CVAT project name")
    parser.add_argument("--project-id", type=int, default=None, help="CVAT project ID")
    parser.add_argument(
        "--url",
        default=os.environ.get("FIFTYONE_CVAT_URL"),
        help="CVAT URL (or set FIFTYONE_CVAT_URL)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("FIFTYONE_CVAT_USERNAME"),
        help="CVAT username (or set FIFTYONE_CVAT_USERNAME)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("FIFTYONE_CVAT_PASSWORD"),
        help="CVAT password (or set FIFTYONE_CVAT_PASSWORD)",
    )
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch FiftyOne App after importing annotations",
    )
    args = parser.parse_args()
    if args.task_ids is None and not any([args.project_name, args.project_id]):
        args.task_ids = [6]

    if not args.url:
        raise ValueError(
            "CVAT URL is required. Set FIFTYONE_CVAT_URL or pass --url."
        )
    if not args.username or not args.password:
        raise ValueError(
            "CVAT username/password are required. "
            "Set FIFTYONE_CVAT_USERNAME/FIFTYONE_CVAT_PASSWORD "
            "or pass --username/--password."
        )

    if args.dataset not in fo.list_datasets():
        if args.dataset == "quickstart":
            print("Dataset 'quickstart' not found. Creating it from FiftyOne Zoo...")
            dataset = foz.load_zoo_dataset("quickstart")
        else:
            raise ValueError(
                f"Dataset '{args.dataset}' not found. "
                "Create/import the dataset first so filenames can be matched."
            )
    else:
        dataset = fo.load_dataset(args.dataset)

    if not any([args.task_ids, args.project_name, args.project_id]):
        raise ValueError(
            "Specify one of --task-id, --project-name, or --project-id."
        )

    fouc.import_annotations(
        dataset,
        task_ids=args.task_ids,
        project_name=args.project_name,
        project_id=args.project_id,
        backend="cvat",
        url=args.url,
        username=args.username,
        password=args.password,
    )
    dataset.save()

    print(
        f"Imported CVAT annotations into dataset='{args.dataset}', "
        f"task_ids={args.task_ids}"
    )
    if args.launch_app:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
