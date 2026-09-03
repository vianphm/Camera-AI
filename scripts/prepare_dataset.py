"""Prepare leak-free train/validation/test dataset splits using subject-based grouping."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def split_dataset(manifest_file: Path, output_dir: Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> None:
    if not manifest_file.exists():
        print(f"[Warning] Manifest {manifest_file} does not exist. Creating default split template.")
        # Create template manifest
        manifest_data = {
            "entries": [
                {"video_id": "vid_001", "subject_id": "subj_1", "action": "walking", "sequence_path": "data/poses/vid_001.npz"},
                {"video_id": "vid_002", "subject_id": "subj_1", "action": "sitting", "sequence_path": "data/poses/vid_002.npz"},
                {"video_id": "vid_003", "subject_id": "subj_2", "action": "falling", "sequence_path": "data/poses/vid_003.npz"},
                {"video_id": "vid_004", "subject_id": "subj_3", "action": "immobile", "sequence_path": "data/poses/vid_004.npz"},
                {"video_id": "vid_005", "subject_id": "subj_4", "action": "stumbling", "sequence_path": "data/poses/vid_005.npz"},
            ]
        }
    else:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    entries = manifest_data.get("entries", [])
    # Group by subject_id to ensure no subject appears across multiple splits
    subjects = list(set(e.get("subject_id", "unknown") for e in entries))
    subjects.sort()

    n_subj = len(subjects)
    n_train = max(1, int(n_subj * train_ratio))
    n_val = max(1, int(n_subj * val_ratio))

    train_subjs = set(subjects[:n_train])
    val_subjs = set(subjects[n_train:n_train + n_val])
    test_subjs = set(subjects[n_train + n_val:])

    train_entries = [e for e in entries if e.get("subject_id") in train_subjs]
    val_entries = [e for e in entries if e.get("subject_id") in val_subjs]
    test_entries = [e for e in entries if e.get("subject_id") in test_subjs]

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    write_jsonl(output_dir / "train.jsonl", train_entries)
    write_jsonl(output_dir / "val.jsonl", val_entries)
    write_jsonl(output_dir / "test.jsonl", test_entries)

    print(f"[Success] Leak-free splits created in {output_dir}:")
    print(f"  • Train: {len(train_entries)} samples ({len(train_subjs)} subjects)")
    print(f"  • Val:   {len(val_entries)} samples ({len(val_subjs)} subjects)")
    print(f"  • Test:  {len(test_entries)} samples ({len(test_subjs)} subjects)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leak-free dataset splits")
    parser.add_argument("--manifest", type=str, default="data/annotations/dataset_manifest.json")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    args = parser.parse_args()

    split_dataset(Path(args.manifest), Path(args.output_dir))


if __name__ == "__main__":
    main()
