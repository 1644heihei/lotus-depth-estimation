import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import HfApi


def _scene_name(rgb_file: str) -> str:
    parts = rgb_file.split("/")
    return parts[1] if len(parts) > 2 else ""


def _index_cache_path(
    repo_id: str,
    split: str,
    max_pairs: int,
    max_scenes: int,
    scene_prefixes: List[str],
    cache_dir: str,
) -> Path:
    scene_key = "-".join(scene_prefixes) if scene_prefixes else "all"
    filename = (
        f"{repo_id.replace('/', '__')}__{split}__mp{max_pairs}__ms{max_scenes}__sc{scene_key}.json"
    )
    return Path(cache_dir) / filename


def _load_cached_pairs(cache_path: Path) -> Optional[List[Tuple[str, str]]]:
    if not cache_path.is_file():
        return None
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [tuple(pair) for pair in payload["pairs"]]


def _save_cached_pairs(cache_path: Path, pairs: List[Tuple[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"pairs": pairs}, f)


def collect_hypersim_rgb_depth_pairs(
    repo_id: str,
    split: str,
    max_pairs: int = 0,
    max_scenes: int = 0,
    scene_prefixes: Optional[List[str]] = None,
    cache_dir: str = "data/.hf_hypersim_index",
    use_cache: bool = True,
) -> List[Tuple[str, str]]:
    """
    Collect rgb/depth file pairs from a Hugging Face Hypersim-Processed dataset.

    Results are cached under cache_dir so subsequent runs skip the remote listing.
    When max_pairs > 0, listing stops as soon as enough pairs are found.
    """
    scene_prefixes = scene_prefixes or []
    cache_path = _index_cache_path(repo_id, split, max_pairs, max_scenes, scene_prefixes, cache_dir)
    if use_cache:
        cached = _load_cached_pairs(cache_path)
        if cached is not None:
            print(f"[hypersim_hf_index] loaded {len(cached)} pairs from cache: {cache_path}")
            return cached

    prefix = f"{split}/"
    items: List[Tuple[str, str]] = []
    api = HfApi()
    print(f"[hypersim_hf_index] scanning {repo_id} ({split}) from Hugging Face Hub...")

    for entry in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True):
        path = entry.path
        if not (path.startswith(prefix) and "/rgb_cam_" in path and path.endswith(".png")):
            continue
        scene = _scene_name(path)
        if scene_prefixes and not any(scene.startswith(prefix_name) for prefix_name in scene_prefixes):
            continue
        depth_file = path.replace("/rgb_cam_", "/depth_plane_cam_")
        items.append((path, depth_file))
        if max_pairs > 0 and len(items) >= max_pairs:
            break

    items.sort()
    if max_scenes > 0:
        chosen_scenes: List[str] = []
        for rgb_file, _ in items:
            scene = _scene_name(rgb_file)
            if scene not in chosen_scenes:
                chosen_scenes.append(scene)
            if len(chosen_scenes) >= max_scenes:
                break
        allowed = set(chosen_scenes)
        items = [(rgb_file, depth_file) for rgb_file, depth_file in items if _scene_name(rgb_file) in allowed]
    if max_pairs > 0:
        items = items[:max_pairs]

    if not items:
        raise ValueError(f"No RGB/depth pairs found in {repo_id}:{split}")

    if use_cache:
        _save_cached_pairs(cache_path, items)
        print(f"[hypersim_hf_index] cached {len(items)} pairs to {cache_path}")

    return items
