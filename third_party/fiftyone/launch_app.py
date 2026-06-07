#!/usr/bin/env python3
"""Launch FiftyOne App with the mmplatform CVAT k-center plugin enabled."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fiftyone as fo  # noqa: E402
import fiftyone.zoo as foz  # noqa: E402
from plugin_config import configure_plugins, make_cvat_panel_spaces  # noqa: E402


def _load_dataset(name: str, max_samples: Optional[int]) -> fo.Dataset:
    if name in fo.list_datasets():
        return fo.load_dataset(name)

    if name == "quickstart":
        kwargs = {"dataset_name": name}
        if max_samples is not None:
            kwargs["max_samples"] = max_samples
        return foz.load_zoo_dataset("quickstart", **kwargs)

    raise ValueError(
        f"Dataset '{name}' not found. Create/import it first, or use --dataset quickstart."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="quickstart", help="FiftyOne dataset name")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only used when creating quickstart from zoo",
    )
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--address", default="0.0.0.0")
    parser.add_argument(
        "--no-auto-panel",
        action="store_true",
        help="Do not open the CVAT k-center panel automatically at startup",
    )
    args = parser.parse_args()

    plugins_dir = configure_plugins()
    dataset = _load_dataset(args.dataset, args.max_samples)
    spaces = None if args.no_auto_panel else make_cvat_panel_spaces()

    print(f"FIFTYONE_PLUGINS_DIR={os.environ.get('FIFTYONE_PLUGINS_DIR')}")
    print("If a FiftyOne server is already running on this port, stop it first (Ctrl+C).")

    session = fo.launch_app(
        dataset,
        port=args.port,
        address=args.address,
        spaces=spaces,
    )
    print(f"FiftyOne App: http://localhost:{args.port}")
    print(f"Plugins dir: {plugins_dir}")
    if args.no_auto_panel:
        print("Open panel manually: Samples タブ横の + → CVAT: k-center 選定")
    else:
        print("CVAT k-center panel should open beside Samples automatically.")
    print("Fallback: Operator browser → Open CVAT k-center panel")
    session.wait()


if __name__ == "__main__":
    main()
