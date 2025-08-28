#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def find_pairs(images_dir: Path, ann_dir: Path) -> list[str]:
    image_exts = {".jpg", ".jpeg", ".png"}
    pairs: list[str] = []
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue
        stem = img_path.stem
        xml = ann_dir / f"{stem}.xml"
        if xml.exists():
            pairs.append(stem)
    return sorted(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Path to images directory")
    ap.add_argument("--annotations", required=True, help="Path to VOC XML directory")
    ap.add_argument("--out_dir", required=True, help="Output dir for train.txt/val.txt")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = Path(args.images)
    anns = Path(args.annotations)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stems = find_pairs(images, anns)
    if not stems:
        raise SystemExit("No image-annotation pairs found. Check paths.")

    rng = random.Random(args.seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * args.val_ratio))
    val = stems[:n_val]
    train = stems[n_val:]

    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    print(f"Wrote {len(train)} train and {len(val)} val names to {out}")


if __name__ == "__main__":
    main()


