#!/usr/bin/env python3
"""Build a portable RGB-focused dataset pack for the enabled Pi3 datasets.

The pack preserves each dataset's expected relative layout while keeping only
the files current RGB-only dataset loaders need: images plus split/camera
metadata. By default files are hard-linked, so the pack is cheap to create on
the same filesystem and still archives as regular file contents with tar.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG", ".JPEG"}

DATASETS = {
    "co3dv2": ("co3dv2_processed", "co3dv2_processed"),
    "wildrgbd": ("wildrgbd_processed", "wildrgbd_processed"),
    "matrixcity": ("matrixcity_processed", "matrixcity_processed"),
    "midair": ("MidAir", "MidAir"),
    "hypersim": ("hypersim_processed", "hypersim_processed"),
    "ntu": ("ntu_seq", "ntu_seq"),
    "real": ("real", "real"),
    "kitti": ("kitti360", "kitti360"),
    "waymo": ("waymo", "waymo"),
    "360_v2": ("360_v2", "360_v2"),
}


class PackStats:
    def __init__(self) -> None:
        self.linked = 0
        self.skipped_existing = 0
        self.missing_roots: list[str] = []
        self.bytes_seen = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "linked": self.linked,
            "skipped_existing": self.skipped_existing,
            "missing_roots": self.missing_roots,
            "bytes_seen": self.bytes_seen,
        }


def is_image(path: Path) -> bool:
    return path.suffix in IMAGE_EXTS or path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def link_file(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    if not src.is_file():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    stats.bytes_seen += size

    if dst.exists():
        if dst.stat().st_size == size:
            stats.skipped_existing += 1
            return
        raise FileExistsError(f"Destination exists with different size: {dst}")

    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    stats.linked += 1


def link_tree(src_root: Path, dst_root: Path, predicate, mode: str, stats: PackStats) -> None:
    if not src_root.exists():
        stats.missing_roots.append(str(src_root))
        return

    for root, _, files in os.walk(src_root):
        root_path = Path(root)
        for name in files:
            src = root_path / name
            rel = src.relative_to(src_root)
            if predicate(rel, src):
                link_file(src, dst_root / rel, mode, stats)


def pack_co3dv2(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        name = src_file.name
        parts = rel.parts
        if name.startswith("selected_seqs_") and name.endswith(".json"):
            return True
        return "images" in parts and (is_image(src_file) or src_file.suffix == ".npz")

    link_tree(src, dst, keep, mode, stats)


def pack_wildrgbd(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        name = src_file.name
        parts = rel.parts
        if name.startswith("selected_seqs_") and name.endswith(".json"):
            return True
        return ("rgb" in parts and is_image(src_file)) or (
            "metadata" in parts and src_file.suffix == ".npz"
        )

    link_tree(src, dst, keep, mode, stats)


def pack_matrixcity(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        parts = rel.parts
        return ("rgb" in parts and is_image(src_file)) or (
            "metadata" in parts and src_file.suffix == ".npz"
        )

    link_tree(src, dst, keep, mode, stats)


def pack_midair(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        parts = rel.parts
        return ("color_left" in parts and is_image(src_file)) or (
            "metadata" in parts and src_file.suffix == ".npz"
        )

    link_tree(src, dst, keep, mode, stats)


def pack_hypersim(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        name = src_file.name
        return (
            name.startswith("cached_metadata_hypersim_") and name.endswith(".h5")
        ) or name.endswith("_rgb.png") or name.endswith("_cam.npz")

    link_tree(src, dst, keep, mode, stats)


def pack_ntu(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        parts = rel.parts
        return src_file.name == "camera_data.npz" or ("rgb" in parts and is_image(src_file))

    link_tree(src, dst, keep, mode, stats)


def pack_real(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        parts = rel.parts
        return ("frames" in parts and is_image(src_file)) or (
            "metadata_json" in parts and src_file.suffix == ".json"
        )

    link_tree(src, dst, keep, mode, stats)


def pack_kitti(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        parts = rel.parts
        return (
            "data_2d_raw" in parts
            and "data_rect" in parts
            and any(part.startswith("image_") for part in parts)
            and is_image(src_file)
        )

    link_tree(src, dst, keep, mode, stats)


def pack_waymo(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    def keep(rel: Path, src_file: Path) -> bool:
        return "exported_images" in rel.parts and is_image(src_file)

    link_tree(src, dst, keep, mode, stats)


def pack_360_v2(src: Path, dst: Path, mode: str, stats: PackStats) -> None:
    image_dirs = {"images", "images_2", "images_4", "images_8"}

    def keep(rel: Path, src_file: Path) -> bool:
        parts = rel.parts
        if len(parts) >= 3 and parts[-3:] in (("sparse", "0", "cameras.bin"), ("sparse", "0", "images.bin")):
            return True
        return any(part in image_dirs for part in parts) and is_image(src_file)

    link_tree(src, dst, keep, mode, stats)


PACKERS = {
    "co3dv2": pack_co3dv2,
    "wildrgbd": pack_wildrgbd,
    "matrixcity": pack_matrixcity,
    "midair": pack_midair,
    "hypersim": pack_hypersim,
    "ntu": pack_ntu,
    "real": pack_real,
    "kitti": pack_kitti,
    "waymo": pack_waymo,
    "360_v2": pack_360_v2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/data/liuwei/dataset")
    parser.add_argument("--dest-root", default="/data/liuwei/dataset/pi3_rgb_pack")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Enabled datasets to include from configs/data/example.yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="hardlink is fastest and keeps the pack archivable without duplicate disk use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    dest_root = Path(args.dest_root).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "mode": args.mode,
        "datasets": {},
        "notes": [
            "This pack intentionally keeps RGB images plus loader-required camera/split metadata.",
            "Depth, masks, point clouds, and unrelated raw files are omitted for the current RGB-only loaders.",
        ],
    }

    total = PackStats()
    for dataset_name in args.datasets:
        src_name, dst_name = DATASETS[dataset_name]
        src = source_root / src_name
        dst = dest_root / dst_name
        stats = PackStats()
        print(f"[{dataset_name}] {src} -> {dst}", flush=True)
        PACKERS[dataset_name](src, dst, args.mode, stats)
        manifest["datasets"][dataset_name] = {
            "source": str(src),
            "destination": str(dst),
            **stats.as_dict(),
        }
        total.linked += stats.linked
        total.skipped_existing += stats.skipped_existing
        total.bytes_seen += stats.bytes_seen
        total.missing_roots.extend(stats.missing_roots)
        print(
            f"[{dataset_name}] linked={stats.linked} skipped={stats.skipped_existing} "
            f"bytes_seen={stats.bytes_seen}",
            flush=True,
        )

    manifest["total"] = total.as_dict()
    manifest_path = dest_root / "PACK_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    print(
        f"Total linked={total.linked} skipped={total.skipped_existing} "
        f"bytes_seen={total.bytes_seen}",
        flush=True,
    )
    if total.missing_roots:
        print("Missing roots:")
        for path in total.missing_roots:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
