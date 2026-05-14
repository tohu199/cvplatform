"""Split a CVAT/COCO export folder into train/val with COCO JSON per split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read a COCO-style folder (annotations/*.json, images/**) and write "
            "images/train, images/val plus annotations/instances_train.json and "
            "annotations/instances_val.json."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="DIR",
        help="Source export directory (e.g. data/exports/task_1_coco)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory to create (e.g. data/exports/task_1_coco_train_val)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        metavar="R",
        help="Fraction of images for train (default: %(default)s); val gets the rest",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed for shuffling image order before split (default: %(default)s)",
    )
    args = p.parse_args()
    args.input = args.input.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be strictly between 0 and 1")
    return args


def _load_coco_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise SystemExit(f"{path}: missing {key!r} (not a COCO dict?)")
    return data


def _merge_instance_jsons(paths: List[Path]) -> Tuple[Dict[str, Any], List[Path]]:
    if not paths:
        raise SystemExit("No instances*.json found under annotations/")
    merged: Dict[str, Any] | None = None
    image_ids: set[int] = set()
    for p in paths:
        d = _load_coco_json(p)
        if merged is None:
            merged = {
                "licenses": d.get("licenses", []),
                "info": d.get("info", {}),
                "categories": d["categories"],
                "images": [],
                "annotations": [],
            }
        else:
            if d["categories"] != merged["categories"]:
                raise SystemExit(
                    f"{p}: categories differ from {paths[0]}; merge not supported."
                )
        for img in d["images"]:
            iid = int(img["id"])
            if iid in image_ids:
                raise SystemExit(f"Duplicate image id {iid} in {p} (already seen)")
            image_ids.add(iid)
            merged["images"].append(img)
        merged["annotations"].extend(d["annotations"])
    assert merged is not None
    return merged, paths


def _basename_to_image_path(images_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not images_root.is_dir():
        raise SystemExit(f"Missing images directory: {images_root}")
    for f in images_root.rglob("*"):
        if not f.is_file():
            continue
        name = f.name
        if name in out:
            raise SystemExit(
                f"Duplicate image file name {name!r} under {images_root}:\n"
                f"  {out[name]}\n  {f}"
            )
        out[name] = f
    return out


def _split_image_indices(n: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if n < 2:
        raise SystemExit(f"Need at least 2 images to split; found {n}")
    n_train = int(n * train_ratio)
    n_train = min(max(n_train, 1), n - 1)
    order = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(order)
    train_idx = set(order[:n_train])
    train_i = [i for i in range(n) if i in train_idx]
    val_i = [i for i in range(n) if i not in train_idx]
    return train_i, val_i


def _write_split(
    base: Dict[str, Any],
    images: List[Dict[str, Any]],
    ann_keep_ids: set[int],
    out_ann: Path,
    images_out_dir: Path,
    name_to_src: Dict[str, Path],
) -> None:
    id_map = {old["id"]: new for new, old in enumerate(images, start=1)}
    new_images: List[Dict[str, Any]] = []
    for im in images:
        row = dict(im)
        row["id"] = id_map[int(im["id"])]
        new_images.append(row)
    new_anns: List[Dict[str, Any]] = []
    next_aid = 1
    for a in base["annotations"]:
        if int(a["image_id"]) not in ann_keep_ids:
            continue
        row = dict(a)
        row["id"] = next_aid
        next_aid += 1
        row["image_id"] = id_map[int(a["image_id"])]
        new_anns.append(row)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "licenses": base.get("licenses", []),
        "info": base.get("info", {}),
        "categories": base["categories"],
        "images": new_images,
        "annotations": new_anns,
    }
    with out_ann.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        fname = im["file_name"]
        src = name_to_src.get(fname)
        if src is None:
            raise SystemExit(
                f"No file matching COCO file_name {fname!r} under images/ "
                f"(expected basename present once)"
            )
        dst = images_out_dir / fname
        shutil.copy2(src, dst)


def main() -> None:
    args = _parse_args()
    ann_dir = args.input / "annotations"
    if not ann_dir.is_dir():
        raise SystemExit(f"Missing annotations directory: {ann_dir}")
    json_paths = sorted(ann_dir.glob("instances*.json"))
    if not json_paths:
        json_paths = sorted(ann_dir.glob("*.json"))
    coco, used_jsons = _merge_instance_jsons(json_paths)
    name_to_src = _basename_to_image_path(args.input / "images")
    images = coco["images"]
    for im in images:
        fn = im.get("file_name")
        if not fn:
            raise SystemExit(f"Image entry missing file_name: {im!r}")
        if fn not in name_to_src:
            raise SystemExit(
                f"COCO file_name {fn!r} not found under {args.input / 'images'}"
            )
    train_idx, val_idx = _split_image_indices(len(images), args.train_ratio, args.seed)
    train_images = [images[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    train_old_ids = {int(im["id"]) for im in train_images}
    val_old_ids = {int(im["id"]) for im in val_images}

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing --out: {args.out}")

    args.out.mkdir(parents=True, exist_ok=False)
    _write_split(
        coco,
        train_images,
        train_old_ids,
        args.out / "annotations" / "instances_train.json",
        args.out / "images" / "train",
        name_to_src,
    )
    _write_split(
        coco,
        val_images,
        val_old_ids,
        args.out / "annotations" / "instances_val.json",
        args.out / "images" / "val",
        name_to_src,
    )
    print(args.out)
    print(f"  merged from: {[p.name for p in used_jsons]}")
    print(f"  train images: {len(train_images)}, val images: {len(val_images)}")


if __name__ == "__main__":
    main()

"""
python3 sdk/split_coco_train_val.py \
  --input data/exports/task_1_coco \
  --out data/exports/task_1_coco_train_val \
  --train-ratio 0.8
  
"""