import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download library-hosted datasets for Lotus training."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="omrastogi/Hypersim-Processed",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hypersim_processed",
        help="Destination directory for downloaded files.",
    )
    parser.add_argument(
        "--include_normals",
        action="store_true",
        help="Also download normal files when available in the source dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        f"{args.split}/**/rgb_cam_*.png",
        f"{args.split}/**/depth_plane_cam_*.png",
    ]
    if args.include_normals:
        allow_patterns.append(f"{args.split}/**/*normal*.png")

    print(f"[prepare_hf_data] repo={args.repo_id} split={args.split}")
    print(f"[prepare_hf_data] output={output_dir}")
    print(f"[prepare_hf_data] patterns={allow_patterns}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    rgb_count = len(list(output_dir.glob(f"{args.split}/**/rgb_cam_*.png")))
    depth_count = len(list(output_dir.glob(f"{args.split}/**/depth_plane_cam_*.png")))
    print(f"[prepare_hf_data] done: rgb={rgb_count}, depth={depth_count}")


if __name__ == "__main__":
    main()
