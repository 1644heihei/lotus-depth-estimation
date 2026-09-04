"""A held-out slice of Hypersim, split by scene, for the text-conditioning experiment.

Training runs on Hypersim and the headline evaluation is NYUv2, which leaves a result
unreadable in one direction: if class-name conditioning does nothing on NYUv2, that could
mean text does not help, or it could mean synthetic-to-real transfer ate the effect. A
held-out slice of the training domain separates those.

Split by SCENE, never by frame. Hypersim frames within one scene are the same room from
nearby camera poses, so a frame-level split leaves near-duplicates on both sides and the
"held-out" set measures memorisation instead of generalisation. This is the same trap the
tilt-predictability regression hit, which is why that one needed GroupKFold.

The split is written to a JSON file and read by both the training script (to exclude those
scenes) and the evaluation (to select them), so the two can never disagree about which
frames were seen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_SPLIT_PATH = Path("datasets/hypersim_holdout.json")
SCENE_RE = re.compile(r"(ai_\d+_\d+)")


def scene_of(image_path: str | Path) -> str | None:
    """Scene id (ai_XXX_YYY) from any path inside the Hypersim tree."""
    m = SCENE_RE.search(str(image_path).replace("\\", "/"))
    return m.group(1) if m else None


def build_split(
    data_root: str | Path,
    n_scenes: int = 20,
    n_eval_frames: int = 300,
    seed: int = 42,
    rgb_glob: str = "rgb_*.png",
) -> dict:
    """Hold out whole scenes, then subsample frames across them for measurement.

    Two separate quantities, because they answer to different pressures. Training must
    exclude EVERY frame of a held-out scene or the split leaks. Evaluation wants breadth
    over depth: 300 frames drawn from 20 rooms says more than 300 frames from 3, where the
    result would ride on whatever those three rooms happen to contain. A first attempt
    took whole scenes until the frame budget filled and got 400 frames from 3 scenes.
    """
    import numpy as np

    root = Path(data_root)
    frames: dict[str, list[str]] = {}
    for p in sorted(root.rglob(rgb_glob)):
        scene = scene_of(p)
        if scene:
            frames.setdefault(scene, []).append(str(p.relative_to(root)).replace("\\", "/"))

    rng = np.random.default_rng(seed)
    order = list(frames)
    rng.shuffle(order)
    held = sorted(order[:n_scenes])

    # spread the eval budget evenly, then even out the remainder over the larger scenes
    per = max(1, n_eval_frames // max(len(held), 1))
    eval_frames: list[str] = []
    for scene in held:
        fs = frames[scene]
        take = min(per, len(fs))
        step = max(1, len(fs) // take)
        eval_frames.extend(fs[::step][:take])
    eval_frames = sorted(eval_frames)[:n_eval_frames]

    return {
        "seed": seed,
        "data_root": str(root),
        "n_scenes_total": len(frames),
        "held_out_scenes": held,
        "n_held_out_scenes": len(held),
        "n_held_out_frames_total": sum(len(frames[s]) for s in held),
        "n_eval_frames": len(eval_frames),
        "eval_frames": eval_frames,
    }


def load_split(path: str | Path = DEFAULT_SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def held_out_scenes(path: str | Path = DEFAULT_SPLIT_PATH) -> set[str]:
    return set(load_split(path)["held_out_scenes"])


def is_held_out(image_path: str | Path, scenes: set[str]) -> bool:
    """True when this frame belongs to a held-out scene; training must skip it."""
    scene = scene_of(image_path)
    return scene is not None and scene in scenes


def main():
    import argparse

    p = argparse.ArgumentParser(description="Build the Hypersim hold-out split.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/hypersim_processed/train")
    p.add_argument("--out", type=str, default=str(DEFAULT_SPLIT_PATH))
    p.add_argument("--n_scenes", type=int, default=20,
                   help="Scenes excluded from training in full.")
    p.add_argument("--n_eval_frames", type=int, default=300,
                   help="Frames sampled across those scenes for measurement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rgb_glob", type=str, default="rgb_*.png")
    args = p.parse_args()

    split = build_split(args.data_root, args.n_scenes, args.n_eval_frames,
                        args.seed, args.rgb_glob)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(f"scenes total        : {split['n_scenes_total']}")
    print(f"held out of training: {split['n_held_out_scenes']} scenes, "
          f"{split['n_held_out_frames_total']} frames")
    print(f"evaluated on        : {split['n_eval_frames']} frames "
          f"(target {args.n_eval_frames})")
    print(f"first few scenes    : {split['held_out_scenes'][:6]}")
    print(f"saved               : {out}")


if __name__ == "__main__":
    main()
