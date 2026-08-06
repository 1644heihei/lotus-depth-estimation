#!/usr/bin/env python
"""Materialize the fixed ScanNet depth-evaluation split from its tar archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split_list", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def normalized_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return str(path).removeprefix("./")


def main():
    args = parse_args()
    pairs = []
    for line in args.split_list.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rgb_rel, depth_rel = line.split()
            pairs.append((rgb_rel, depth_rel))

    wanted = {rel for pair in pairs for rel in pair}
    written = set()
    with tarfile.open(args.archive, "r:*") as archive:
        members = {
            normalized_member(member.name): member
            for member in archive.getmembers()
            if member.isfile()
        }
        missing = wanted.difference(members)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} split files are missing from {args.archive}; "
                f"first={sorted(missing)[0]}"
            )
        for rel in sorted(wanted):
            source = archive.extractfile(members[rel])
            if source is None:
                raise OSError(f"Could not read {rel} from {args.archive}")
            destination = args.output_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
            written.add(rel)

    manifest = {
        "dataset": "scannet_depth_eval",
        "depth_unit": "mm",
        "source_archive": str(args.archive),
        "split_list": str(args.split_list),
        "num_images": len(pairs),
        "num_files": len(written),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
