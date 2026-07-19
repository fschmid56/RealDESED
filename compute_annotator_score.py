import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_FRAME_HZ = 25
DEFAULT_MIN_FILES = 1
DEFAULT_SHRINKAGE_K = 5.0
DEFAULT_DATA_ROOT = Path("data")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def deduplicate_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduplicated = []

    for row in rows:
        key = tuple((column, row.get(column, "")) for column in row.keys())
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)

    return deduplicated


def prepare_train_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    by_file = defaultdict(list)

    for row in rows:
        if row.get("review_status", "").strip().lower() != "unreviewed":
            continue
        by_file[row["filename"]].append(row)

    selected = []
    for file_rows in by_file.values():
        selected.extend(file_rows)

    return selected


def group_events_by_file(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, object]]]:
    grouped = defaultdict(list)

    for row in rows:
        label = row.get("class", "").strip()
        annotator_id = row.get("annotator_id", "").strip()

        if not label or not annotator_id:
            continue

        start = float(row["onset"])
        end = float(row["offset"])
        if end <= start:
            continue

        grouped[row["filename"]].append(
            {
                "start": start,
                "end": end,
                "label": label,
                "annotator_id": annotator_id,
            }
        )

    return dict(grouped)


def class_names_from_events(grouped_events: Dict[str, List[Dict[str, object]]]) -> List[str]:
    return sorted(
        {
            str(event["label"])
            for events in grouped_events.values()
            for event in events
        }
    )


def events_to_frame_labels(
    events: Sequence[Dict[str, object]],
    class_to_idx: Dict[str, int],
    duration: float,
    frame_hz: int,
) -> torch.Tensor:
    n_frames = int(math.ceil(duration * frame_hz))
    labels = torch.zeros(len(class_to_idx), n_frames, dtype=torch.float32)

    for event in events:
        label = str(event["label"])
        if label not in class_to_idx:
            continue

        start_frame = max(0, int(math.ceil(float(event["start"]) * frame_hz - 0.5)))
        end_frame = min(n_frames, int(math.ceil(float(event["end"]) * frame_hz - 0.5)))
        if end_frame <= start_frame:
            continue

        labels[class_to_idx[label], start_frame:end_frame] = 1.0

    return labels


def macro_f1_dice(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    active_only: bool = True,
) -> Optional[float]:
    intersection = (prediction * reference).sum(dim=1)
    prediction_size = prediction.sum(dim=1)
    reference_size = reference.sum(dim=1)
    denominator = prediction_size + reference_size

    if active_only:
        valid = denominator > 0
    else:
        valid = torch.ones_like(denominator, dtype=torch.bool)

    if valid.sum().item() == 0:
        return None

    per_class = torch.zeros_like(denominator)
    active = denominator > 0
    per_class[active] = 2.0 * intersection[active] / denominator[active].clamp_min(1e-8)
    per_class[~active] = 1.0

    return per_class[valid].mean().item()


def group_by_annotator(
    events: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event["annotator_id"])].append(event)
    return dict(grouped)


def score_file_pairwise_macro_f1(
    events: Sequence[Dict[str, object]],
    class_to_idx: Dict[str, int],
    frame_hz: int,
) -> Dict[str, Tuple[float, int]]:
    by_annotator = group_by_annotator(events)
    if len(by_annotator) < 2:
        return {}

    duration = max(float(event["end"]) for event in events)
    labels_by_annotator = {
        annotator_id: events_to_frame_labels(ann_events, class_to_idx, duration, frame_hz)
        for annotator_id, ann_events in by_annotator.items()
    }

    file_scores = {}
    for annotator_id, labels in labels_by_annotator.items():
        pairwise_scores = []
        for other_id, other_labels in labels_by_annotator.items():
            if other_id == annotator_id:
                continue

            score = macro_f1_dice(labels, other_labels, active_only=True)
            if score is not None:
                pairwise_scores.append(score)

        if pairwise_scores:
            file_scores[annotator_id] = (
                sum(pairwise_scores) / len(pairwise_scores),
                len(pairwise_scores),
            )

    return file_scores


def score_annotators(
    grouped_events: Dict[str, List[Dict[str, object]]],
    class_names: Sequence[str],
    frame_hz: int,
    min_files: int,
    shrinkage_k: float,
) -> List[Dict[str, object]]:
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    file_scores_by_annotator = defaultdict(list)
    comparisons_by_annotator = defaultdict(int)

    for events in grouped_events.values():
        file_scores = score_file_pairwise_macro_f1(events, class_to_idx, frame_hz)
        for annotator_id, (score, num_comparisons) in file_scores.items():
            file_scores_by_annotator[annotator_id].append(score)
            comparisons_by_annotator[annotator_id] += num_comparisons

    raw_results = []
    for annotator_id, scores in file_scores_by_annotator.items():
        if len(scores) < min_files:
            continue

        raw_results.append(
            {
                "annotator_id": annotator_id,
                "raw_score": sum(scores) / len(scores),
                "num_files": len(scores),
                "num_pairwise_comparisons": comparisons_by_annotator[annotator_id],
            }
        )

    if not raw_results:
        return []

    global_mean = sum(result["raw_score"] for result in raw_results) / len(raw_results)
    results = []
    for result in raw_results:
        confidence = result["num_files"] / (result["num_files"] + shrinkage_k)
        score = confidence * result["raw_score"] + (1.0 - confidence) * global_mean
        results.append(
            {
                "annotator_id": result["annotator_id"],
                "score": score,
                "raw_score": result["raw_score"],
                "confidence": confidence,
                "num_files": result["num_files"],
                "num_pairwise_comparisons": result["num_pairwise_comparisons"],
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)


def write_json(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(list(rows), handle, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotator_id",
        "score",
        "raw_score",
        "confidence",
        "num_files",
        "num_pairwise_comparisons",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_annotations_path(root: Path, split: str) -> Path:
    split_dir = root / split
    filename = "annotations_raw.csv" if split == "train" else "annotations.csv"
    return split_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate annotator reliability with pairwise frame-level macro F1 "
            "(Dice coefficient) and shrink scores toward the global mean."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/validation/test subdirectories.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional explicit path to an annotations CSV file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split used when --annotations is not provided.",
    )
    parser.add_argument("--frame_hz", type=int, default=DEFAULT_FRAME_HZ)
    parser.add_argument(
        "--min_files",
        type=int,
        default=DEFAULT_MIN_FILES,
        help="Minimum number of multi-annotator files required for an annotator score.",
    )
    parser.add_argument(
        "--shrinkage_k",
        type=float,
        default=DEFAULT_SHRINKAGE_K,
        help=(
            "Empirical-Bayes shrinkage strength. An annotator with k files gets "
            "50 percent weight on its raw score and 50 percent on the global mean."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for annotator_scores.{json,csv}. Defaults to <root>/annotator_quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split
    annotations_path = args.annotations or default_annotations_path(args.root, split)
    output_dir = args.output_dir or args.root / "annotator_quality"

    rows = deduplicate_rows(read_csv(annotations_path))
    if annotations_path.name == "annotations_raw.csv":
        rows = prepare_train_rows(rows)

    grouped_events = group_events_by_file(rows)
    class_names = class_names_from_events(grouped_events)
    results = score_annotators(
        grouped_events=grouped_events,
        class_names=class_names,
        frame_hz=args.frame_hz,
        min_files=args.min_files,
        shrinkage_k=args.shrinkage_k,
    )

    json_path = output_dir / f"annotator_scores_{split}.json"
    csv_path = output_dir / f"annotator_scores_{split}.csv"
    write_json(json_path, results)
    write_csv(csv_path, results)

    print(f"Scored {len(results)} annotators from {annotations_path}")
    print(f"Classes: {len(class_names)}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
