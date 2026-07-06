import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download

from utils.hypersim_hf_index import collect_hypersim_rgb_depth_pairs


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
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Maximum number of rgb-depth pairs to download (0 means no limit).",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help="Maximum number of scenes to include (0 means no limit).",
    )
    parser.add_argument(
        "--scene_prefix",
        type=str,
        default="",
        help="Optional comma-separated scene prefixes (e.g. 'ai_001,ai_002').",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.0,
        help="Optional sleep interval between file downloads to reduce API pressure.",
    )
    return parser.parse_args()


def get_scene_name(path: str) -> str:
    # Expected: split/scene_name/filename.png
    parts = path.split("/")
    return parts[1] if len(parts) > 2 else ""


def collect_pairs(
    repo_id: str,
    split: str,
    max_pairs: int,
    max_scenes: int,
    scene_prefixes: List[str],
) -> List[Tuple[str, str]]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    depth_files = set(
        f for f in files if f.startswith(f"{split}/") and "/depth_plane_cam_" in f and f.endswith(".png")
    )

    pairs: List[Tuple[str, str]] = []
    for rgb_file in files:
        if not (rgb_file.startswith(f"{split}/") and "/rgb_cam_" in rgb_file and rgb_file.endswith(".png")):
            continue
        scene = get_scene_name(rgb_file)
        if scene_prefixes and not any(scene.startswith(prefix) for prefix in scene_prefixes):
            continue
        depth_file = rgb_file.replace("/rgb_cam_", "/depth_plane_cam_")
        if depth_file in depth_files:
            pairs.append((rgb_file, depth_file))

    pairs.sort()
    if max_scenes > 0:
        chosen_scenes = []
        for rgb_file, _ in pairs:
            scene = get_scene_name(rgb_file)
            if scene not in chosen_scenes:
                chosen_scenes.append(scene)
            if len(chosen_scenes) >= max_scenes:
                break
        allowed = set(chosen_scenes)
        pairs = [(r, d) for r, d in pairs if get_scene_name(r) in allowed]

    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_prefixes = [x.strip() for x in args.scene_prefix.split(",") if x.strip()]
    print(f"[prepare_hf_data] repo={args.repo_id} split={args.split}")
    print(f"[prepare_hf_data] output={output_dir}")
    print(
        f"[prepare_hf_data] constraints: max_pairs={args.max_pairs or 'all'} "
        f"max_scenes={args.max_scenes or 'all'} scene_prefixes={scene_prefixes or 'all'}"
    )

    pairs = collect_hypersim_rgb_depth_pairs(
        repo_id=args.repo_id,
        split=args.split,
        max_pairs=args.max_pairs,
        max_scenes=args.max_scenes,
        scene_prefixes=scene_prefixes,
        use_cache=True,
    )
    print(f"[prepare_hf_data] selected_pairs={len(pairs)}")

    normals_cache: Dict[str, str] = {}
    for idx, (rgb_file, depth_file) in enumerate(pairs, start=1):
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=rgb_file,
            local_dir=str(output_dir),
        )
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=depth_file,
            local_dir=str(output_dir),
        )

        if args.include_normals:
            normal_file = rgb_file.replace("/rgb_cam_", "/normal_cam_")
            if normal_file not in normals_cache:
                try:
                    hf_hub_download(
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        filename=normal_file,
                        local_dir=str(output_dir),
                    )
                    normals_cache[normal_file] = "ok"
                except Exception:
                    normals_cache[normal_file] = "missing"

        if idx % 50 == 0:
            print(f"[prepare_hf_data] downloaded_pairs={idx}/{len(pairs)}")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    rgb_count = len(list(output_dir.glob(f"{args.split}/**/rgb_cam_*.png")))
    depth_count = len(list(output_dir.glob(f"{args.split}/**/depth_plane_cam_*.png")))
    print(f"[prepare_hf_data] done: rgb={rgb_count}, depth={depth_count}")


if __name__ == "__main__":
    main()
