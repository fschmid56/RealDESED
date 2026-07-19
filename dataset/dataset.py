import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio
import torchaudio.functional as torchaudio_functional


RealDESED_CLASSES = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]


class RealDESEDDataset(torch.utils.data.Dataset):
    """
    Dataset loader for RealDESED.

    Expected layout:
        root/
            train/{audio,metadata.csv,annotations.csv,annotations_raw.csv}
            validation/{audio,metadata.csv,annotations.csv,annotations_raw.csv}
            test/{audio,metadata.csv,annotations.csv,annotations_raw.csv}

    Validation and test use annotations.csv. Train uses
    annotations_raw.csv to support annotator aggregation experiments.
    """

    AGGREGATIONS = {
        "Random (Fixed)",
        "Random (Epoch)",
        "Majority",
        "Intersection",
        "Union",
        "Collector",
        "Uniform Soft",
        "Weighted Soft",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 16000,
        chunk_size: float = 10.0,
        frame_hz: int = 25,
        classes: Optional[Sequence[str]] = None,
        aggregation: str = "Majority",
        annotator_weight_alpha: float = 16.0,
        include_reviewed_train_files: bool = False,
    ):
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Expected one of {sorted(self.AGGREGATIONS)}."
            )

        self.root = Path(root)
        self.split = self._normalize_split(split)
        self.split_dir = self.root / self.split
        self.audio_dir = self.split_dir / "audio"
        self.sample_rate = sample_rate
        self.chunk_size = float(chunk_size)
        self.chunk_samples = int(round(self.chunk_size * self.sample_rate))
        self.frame_hz = frame_hz
        self.frame_resolution = 1.0 / frame_hz
        self.aggregation = aggregation
        self.annotator_weight_alpha = annotator_weight_alpha
        self.default_annotator_weight = 1.0
        self.include_reviewed_train_files = include_reviewed_train_files

        self.metadata = self._read_csv(self.split_dir / "metadata.csv")
        self.metadata_by_file = {row["filename"]: row for row in self.metadata}

        if classes is None:
            classes = RealDESED_CLASSES
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        if self.split == "train":
            raw_rows = self._read_csv(self.split_dir / "annotations_raw.csv")
            raw_rows = self._deduplicate_rows(raw_rows)
            self.raw_annotations = self._prepare_train_raw_annotations(raw_rows)
            self.annotations = self._group_events(self.raw_annotations)
            self.annotator_weights = self._compute_annotator_weights()
            self.fixed_random_annotators = self._select_fixed_random_annotators()
        else:
            rows = self._read_csv(self.split_dir / "annotations.csv")
            self.raw_annotations = []
            self.annotations = self._group_events(rows)
            self.annotator_weights = {}
            self.fixed_random_annotators = {}

        filenames = [row["filename"] for row in self.metadata]
        filenames = [f for f in filenames if f in self.annotations]
        self.files = filenames

    @staticmethod
    def _normalize_split(split: str) -> str:
        if split in {"train", "validation", "test"}:
            return split
        raise ValueError("split must be 'train', 'validation', or 'test'")

    @staticmethod
    def _read_csv(path: Path) -> List[Dict[str, str]]:
        with path.open("r", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _deduplicate_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        deduplicated = []

        for row in rows:
            key = tuple((column, row.get(column, "")) for column in row.keys())
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(row)

        return deduplicated

    def _prepare_train_raw_annotations(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        by_file = defaultdict(list)
        expected_statuses = {"accepted", "rejected", "unreviewed"}
        for row in rows:
            status = row.get("review_status", "").strip().lower()
            if status not in expected_statuses:
                raise ValueError(
                    "Unexpected review_status "
                    f"'{row.get('review_status', '')}' for file {row.get('filename', '<unknown>')}. "
                    f"Expected one of {sorted(expected_statuses)}."
                )
            row["review_status"] = status
            if status == "rejected":
                continue
            by_file[row["filename"]].append(row)

        selected = []
        for filename, file_rows in by_file.items():
            accepted = [row for row in file_rows if row["review_status"] == "accepted"]
            unreviewed = [row for row in file_rows if row["review_status"] == "unreviewed"]

            if self.include_reviewed_train_files and accepted:
                selected.extend(accepted)
            else:
                selected.extend(unreviewed)

        return selected

    @staticmethod
    def _group_events(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, object]]]:
        grouped = defaultdict(list)

        for row in rows:
            label = row.get("class", "").strip()
            event = {
                "start": float(row["onset"]),
                "end": float(row["offset"]),
                "label": label,
            }

            if "annotator_id" in row:
                event["annotator_id"] = row.get("annotator_id", "")
                event["annotator_is_creator"] = row.get("annotator_is_creator", "") == "True"
                event["review_status"] = row.get("review_status", "")
                event["reviewer_id"] = row.get("reviewer_id", "")

            grouped[row["filename"]].append(event)

        return dict(grouped)

    def _load_audio(self, filename: str) -> Tuple[torch.Tensor, float]:
        path = self.audio_dir / filename
        audio, original_rate = torchaudio.load(path)

        if original_rate != self.sample_rate:
            audio = torchaudio_functional.resample(
                audio,
                orig_freq=original_rate,
                new_freq=self.sample_rate,
            )

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        duration = audio.shape[1] / self.sample_rate
        return audio, duration

    def _events_to_labels(
        self,
        events: Sequence[Dict[str, object]],
        start_time: float,
        duration: float,
    ) -> torch.Tensor:
        n_frames = int(duration * self.frame_hz)
        labels = torch.zeros(len(self.classes), n_frames, dtype=torch.float32)
        if n_frames == 0:
            return labels

        segment_end = start_time + duration

        for event in events:
            label = event["label"]
            if label not in self.class_to_idx:
                continue

            event_start = max(float(event["start"]), start_time)
            event_end = min(float(event["end"]), segment_end)
            if event_end <= event_start:
                continue

            first_frame = max(
                0,
                math.ceil((event_start - start_time) * self.frame_hz - 0.5),
            )
            last_frame = min(
                n_frames,
                math.ceil((event_end - start_time) * self.frame_hz - 0.5),
            )
            labels[self.class_to_idx[label], first_frame:last_frame] = 1.0

        return labels

    @staticmethod
    def _majority(stacked: torch.Tensor) -> torch.Tensor:
        return (stacked.mean(dim=0) >= 0.5).float()

    def _group_by_annotator(
        self,
        events: Sequence[Dict[str, object]],
    ) -> Dict[str, List[Dict[str, object]]]:
        grouped = defaultdict(list)
        for event in events:
            grouped[str(event.get("annotator_id", ""))].append(event)
        return dict(grouped)

    def _select_fixed_random_annotators(self) -> Dict[str, str]:
        if self.aggregation != "Random (Fixed)":
            return {}

        selected = {}
        for filename, events in self.annotations.items():
            by_annotator = self._group_by_annotator(events)
            if by_annotator:
                selected[filename] = random.choice(list(by_annotator.keys()))

        return selected

    @staticmethod
    def _contains_accepted_review(events: Sequence[Dict[str, object]]) -> bool:
        return any(event.get("review_status") == "accepted" for event in events)

    def _aggregate_train_events(
        self,
        events: Sequence[Dict[str, object]],
        start_time: float,
        duration: float,
        filename: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not events:
            empty = torch.zeros(len(self.classes), int(duration * self.frame_hz))
            return empty, empty

        by_annotator = self._group_by_annotator(events)

        if self.aggregation == "Random (Fixed)":
            if filename is None:
                raise ValueError("filename is required for Random (Fixed) aggregation")

            annotator_id = self.fixed_random_annotators.get(filename)
            fixed_events = by_annotator.get(annotator_id, [])
            labels = self._events_to_labels(fixed_events, start_time, duration)
            labels = (labels > 0).float()
            return labels, labels

        annotator_ids = list(by_annotator.keys())
        per_annotator = [
            self._events_to_labels(by_annotator[annotator_id], start_time, duration)
            for annotator_id in annotator_ids
        ]
        stacked = torch.stack(per_annotator, dim=0)

        if self.aggregation == "Random (Epoch)":
            labels = per_annotator[random.randrange(len(per_annotator))]
            labels = (labels > 0).float()
            return labels, labels

        if self.aggregation == "Union":
            labels = (stacked.max(dim=0).values > 0).float()
            return labels, labels

        if self.aggregation == "Intersection":
            labels = (stacked.min(dim=0).values > 0).float()
            return labels, labels

        if self.aggregation == "Uniform Soft":
            soft = stacked.mean(dim=0)
            hard = (soft >= 0.5).float()
            return soft, hard

        if self.aggregation == "Weighted Soft":
            if self._contains_accepted_review(events):
                # Accepted reviewed annotations are only loaded when
                # include_reviewed_train_files is enabled. They already passed
                # quality control, so do not reweight them with scores estimated
                # for unreviewed annotators.
                soft = stacked.mean(dim=0)
                hard = (soft >= 0.5).float()
                return soft, hard

            weights = torch.tensor(
                [
                    self.annotator_weights.get(annotator_id, self.default_annotator_weight)
                    for annotator_id in annotator_ids
                ],
                dtype=torch.float32,
            )
            weights = weights.clamp(min=1e-6)
            weights = weights ** self.annotator_weight_alpha
            weights = weights / weights.sum()
            soft = (stacked * weights[:, None, None]).sum(dim=0)
            hard = (soft >= 0.5).float()
            return soft, hard

        if self.aggregation == "Collector":
            creator_events = [
                event for event in events
                if bool(event.get("annotator_is_creator", False))
            ]
            if creator_events:
                labels = self._events_to_labels(creator_events, start_time, duration)
                labels = (labels > 0).float()
                return labels, labels

        labels = self._majority(stacked)
        return labels, labels

    def _compute_annotator_weights(self) -> Dict[str, float]:
        if self.aggregation != "Weighted Soft":
            return {}

        score_path = self.root / "annotator_quality" / "annotator_scores_train.json"
        if not score_path.exists():
            raise FileNotFoundError(
                f"Missing annotator score file for Weighted Soft aggregation: {score_path}. "
                "Run compute_annotator_score.py first."
            )

        with score_path.open("r") as handle:
            results = json.load(handle)

        self.annotator_score_results = results

        weights = {
            str(result["annotator_id"]): float(result["score"])
            for result in results
        }
        if weights:
            self.default_annotator_weight = sum(weights.values()) / len(weights)

        return weights

    def _timestamps(self, duration: float, start_time: float = 0.0) -> torch.Tensor:
        n_frames = int(duration * self.frame_hz)
        return (
            torch.arange(n_frames, dtype=torch.float32) * self.frame_resolution
            + 0.5 * self.frame_resolution
            + start_time
        )

    def _metadata_for_file(self, filename: str) -> Dict[str, object]:
        row = self.metadata_by_file[filename]
        return {
            "filename": filename,
            "target_classes": self._split_list(row.get("target_classes", "")),
            "non_target_classes": self._split_list(row.get("non_target_classes", "")),
            "recording_device": row.get("recording_device", "").strip(),
            "device_placement": row.get("device_placement", "").strip(),
            "recording_environment": self._split_list(row.get("recording_environment", "")),
            "scene_description": row.get("scene_description", ""),
            "license": row.get("license", ""),
        }

    @staticmethod
    def _split_list(value: str) -> List[str]:
        return [item.strip() for item in value.split(";") if item.strip()]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        filename = self.files[index]
        audio, total_duration = self._load_audio(filename)
        events = self.annotations.get(filename, [])
        metadata = self._metadata_for_file(filename)

        if self.split == "train":
            if audio.shape[1] > self.chunk_samples:
                start_sample = random.randint(0, audio.shape[1] - self.chunk_samples)
            else:
                start_sample = 0

            end_sample = start_sample + self.chunk_samples
            audio_chunk = audio[:, start_sample:end_sample]
            if audio_chunk.shape[1] < self.chunk_samples:
                pad = self.chunk_samples - audio_chunk.shape[1]
                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad))

            start_time = start_sample / self.sample_rate
            soft_labels, hard_labels = self._aggregate_train_events(
                events,
                start_time=start_time,
                duration=self.chunk_size,
                filename=filename,
            )

            return {
                "audio": audio_chunk,
                "soft_labels": soft_labels,
                "hard_labels": hard_labels,
                "filename": filename,
                "timestamps": self._timestamps(self.chunk_size, start_time),
                "duration": self.chunk_size,
                "events": events,
                "metadata": metadata,
            }

        labels = self._events_to_labels(events, start_time=0.0, duration=total_duration)

        return {
            "audio": audio,
            "labels": labels,
            "filename": filename,
            "timestamps": self._timestamps(total_duration),
            "duration": total_duration,
            "events": events,
            "metadata": metadata,
        }
