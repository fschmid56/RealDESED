import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataset.dataset as dataset_module
from dataset.collate import collate_fn
from dataset.dataset import RealDESEDDataset


TEST_CLASSES = ["bell_ringing", "coffee_machine"]


class RealDESEDDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.sample_rate = 8000
        self.frame_hz = 10
        self.audio_by_name = {}
        self._write_dataset()
        self.load_patcher = mock.patch(
            "dataset.dataset.torchaudio.load",
            side_effect=self._mock_torchaudio_load,
        )
        self.load_patcher.start()

    def tearDown(self):
        self.load_patcher.stop()
        self.temp_dir.cleanup()

    def _write_dataset(self):
        for split in ("train", "validation", "test"):
            split_dir = self.root / split
            (split_dir / "audio").mkdir(parents=True)
            self._write_metadata(split_dir / "metadata.csv", f"{split}_a.wav")
            self._write_audio(split_dir / "audio" / f"{split}_a.wav", duration=1.2)

        self._write_csv(
            self.root / "train" / "annotations_raw.csv",
            [
                {
                    "filename": "train_a.wav",
                    "onset": "0.10",
                    "offset": "0.40",
                    "class": "bell_ringing",
                    "annotator_id": "ann_good",
                    "annotator_is_creator": "True",
                    "review_status": "unreviewed",
                    "reviewer_id": "",
                },
                {
                    "filename": "train_a.wav",
                    "onset": "0.50",
                    "offset": "0.80",
                    "class": "coffee_machine",
                    "annotator_id": "ann_bad",
                    "annotator_is_creator": "False",
                    "review_status": "unreviewed",
                    "reviewer_id": "",
                },
                {
                    "filename": "train_a.wav",
                    "onset": "0.00",
                    "offset": "1.00",
                    "class": "bell_ringing",
                    "annotator_id": "rejected",
                    "annotator_is_creator": "False",
                    "review_status": "rejected",
                    "reviewer_id": "reviewer",
                },
                {
                    "filename": "train_a.wav",
                    "onset": "0.20",
                    "offset": "0.30",
                    "class": "coffee_machine",
                    "annotator_id": "accepted",
                    "annotator_is_creator": "False",
                    "review_status": "accepted",
                    "reviewer_id": "reviewer",
                },
            ],
        )

        for split in ("validation", "test"):
            self._write_csv(
                self.root / split / "annotations.csv",
                [
                    {
                        "filename": f"{split}_a.wav",
                        "onset": "0.00",
                        "offset": "0.20",
                        "class": "bell_ringing",
                    },
                    {
                        "filename": f"{split}_a.wav",
                        "onset": "0.90",
                        "offset": "1.10",
                        "class": "unknown_class",
                    },
                ],
            )

        quality_dir = self.root / "annotator_quality"
        quality_dir.mkdir()
        with (quality_dir / "annotator_scores_train.json").open("w") as handle:
            json.dump(
                [
                    {"annotator_id": "ann_good", "score": 0.9},
                    {"annotator_id": "ann_bad", "score": 0.1},
                ],
                handle,
            )

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_metadata(path, filename):
        rows = [
            {
                "filename": filename,
                "target_classes": "bell_ringing; coffee_machine",
                "non_target_classes": "footsteps",
                "recording_device": "iPhone 15",
                "device_placement": "Static",
                "recording_environment": "kitchen; hallway",
                "scene_description": "short synthetic clip",
                "license": "test",
            },
            {
                "filename": "no_annotations.wav",
                "target_classes": "bell_ringing",
                "non_target_classes": "",
                "recording_device": "Pixel",
                "device_placement": "Mobile",
                "recording_environment": "bedroom",
                "scene_description": "excluded because there are no annotations",
                "license": "test",
            },
        ]
        RealDESEDDatasetTest._write_csv(path, rows)

    def _write_audio(self, path, duration):
        samples = int(self.sample_rate * duration)
        self.audio_by_name[path.name] = torch.linspace(-0.2, 0.2, samples).repeat(2, 1)
        path.touch()

    def _mock_torchaudio_load(self, path):
        filename = Path(path).name
        return self.audio_by_name[filename].clone(), self.sample_rate

    def _train_dataset(self, aggregation="Majority", include_reviewed=False):
        return RealDESEDDataset(
            root=str(self.root),
            split="train",
            sample_rate=self.sample_rate,
            chunk_size=1.0,
            frame_hz=self.frame_hz,
            classes=TEST_CLASSES,
            aggregation=aggregation,
            annotator_weight_alpha=1.0,
            include_reviewed_train_files=include_reviewed,
        )

    def test_rejects_invalid_split_and_aggregation(self):
        with self.assertRaises(ValueError):
            RealDESEDDataset(root=str(self.root), split="dev")

        with self.assertRaises(ValueError):
            self._train_dataset(aggregation="not-a-mode")

    def test_train_filters_review_status_and_metadata_files(self):
        dataset = self._train_dataset()

        self.assertEqual(dataset.files, ["train_a.wav"])
        self.assertEqual(len(dataset.raw_annotations), 2)
        self.assertNotIn("rejected", {row["annotator_id"] for row in dataset.raw_annotations})

        metadata = dataset._metadata_for_file("train_a.wav")
        self.assertEqual(metadata["target_classes"], TEST_CLASSES)
        self.assertEqual(metadata["non_target_classes"], ["footsteps"])
        self.assertEqual(metadata["recording_environment"], ["kitchen", "hallway"])
        self.assertEqual(metadata["device_placement"], "Static")

    def test_include_reviewed_train_files_prefers_accepted_rows(self):
        dataset = self._train_dataset(include_reviewed=True)

        self.assertEqual(len(dataset.raw_annotations), 1)
        self.assertEqual(dataset.raw_annotations[0]["annotator_id"], "accepted")
        self.assertEqual(dataset.raw_annotations[0]["review_status"], "accepted")

    def test_event_to_label_frame_alignment_and_unknown_classes(self):
        dataset = self._train_dataset()
        labels = dataset._events_to_labels(
            [
                {"start": 0.10, "end": 0.40, "label": "bell_ringing"},
                {"start": 0.10, "end": 0.40, "label": "unknown_class"},
            ],
            start_time=0.0,
            duration=1.0,
        )

        self.assertEqual(labels.shape, (2, 10))
        self.assertTrue(torch.equal(labels[0], torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]).float()))
        self.assertEqual(labels[1].sum().item(), 0.0)

    def test_all_train_aggregation_modes(self):
        events = [
            {
                "start": 0.10,
                "end": 0.40,
                "label": "bell_ringing",
                "annotator_id": "ann_good",
                "annotator_is_creator": True,
                "review_status": "unreviewed",
            },
            {
                "start": 0.50,
                "end": 0.80,
                "label": "coffee_machine",
                "annotator_id": "ann_bad",
                "annotator_is_creator": False,
                "review_status": "unreviewed",
            },
        ]

        union_soft, union_hard = self._train_dataset("Union")._aggregate_train_events(events, 0.0, 1.0)
        self.assertTrue(torch.equal(union_soft, union_hard))
        self.assertEqual(union_hard[0].sum().item(), 3.0)
        self.assertEqual(union_hard[1].sum().item(), 3.0)

        intersection_soft, intersection_hard = self._train_dataset("Intersection")._aggregate_train_events(events, 0.0, 1.0)
        self.assertTrue(torch.equal(intersection_soft, intersection_hard))
        self.assertEqual(intersection_hard.sum().item(), 0.0)

        majority_soft, majority_hard = self._train_dataset("Majority")._aggregate_train_events(events, 0.0, 1.0)
        self.assertTrue(torch.equal(majority_soft, majority_hard))
        self.assertEqual(majority_hard[0].sum().item(), 3.0)
        self.assertEqual(majority_hard[1].sum().item(), 3.0)

        soft, hard = self._train_dataset("Uniform Soft")._aggregate_train_events(events, 0.0, 1.0)
        self.assertAlmostEqual(soft[0, 1].item(), 0.5)
        self.assertAlmostEqual(soft[1, 5].item(), 0.5)
        self.assertEqual(hard[0].sum().item(), 3.0)

        weighted_soft, weighted_hard = self._train_dataset("Weighted Soft")._aggregate_train_events(events, 0.0, 1.0)
        self.assertAlmostEqual(weighted_soft[0, 1].item(), 0.9, places=5)
        self.assertAlmostEqual(weighted_soft[1, 5].item(), 0.1, places=5)
        self.assertEqual(weighted_hard[0].sum().item(), 3.0)
        self.assertEqual(weighted_hard[1].sum().item(), 0.0)

        collector_soft, collector_hard = self._train_dataset("Collector")._aggregate_train_events(events, 0.0, 1.0)
        self.assertTrue(torch.equal(collector_soft, collector_hard))
        self.assertEqual(collector_hard[0].sum().item(), 3.0)
        self.assertEqual(collector_hard[1].sum().item(), 0.0)

        random.seed(7)
        fixed = self._train_dataset("Random (Fixed)")
        fixed_soft, fixed_hard = fixed._aggregate_train_events(events, 0.0, 1.0, filename="train_a.wav")
        self.assertTrue(torch.equal(fixed_soft, fixed_hard))
        self.assertIn(fixed_hard.sum().item(), {3.0})

        epoch_soft, epoch_hard = self._train_dataset("Random (Epoch)")._aggregate_train_events(events, 0.0, 1.0)
        self.assertTrue(torch.equal(epoch_soft, epoch_hard))
        self.assertIn(epoch_hard.sum().item(), {3.0})

    def test_weighted_soft_requires_annotator_scores(self):
        (self.root / "annotator_quality" / "annotator_scores_train.json").unlink()

        with self.assertRaises(FileNotFoundError):
            self._train_dataset("Weighted Soft")

    def test_train_getitem_loads_audio_pads_chunk_and_returns_labels(self):
        dataset = self._train_dataset("Uniform Soft")
        item = dataset[0]

        self.assertEqual(item["audio"].shape, (1, self.sample_rate))
        self.assertEqual(item["soft_labels"].shape, (2, 10))
        self.assertEqual(item["hard_labels"].shape, (2, 10))
        self.assertEqual(item["timestamps"].shape, (10,))
        self.assertEqual(item["filename"], "train_a.wav")
        self.assertEqual(item["duration"], 1.0)
        self.assertEqual(item["metadata"]["recording_device"], "iPhone 15")

    def test_eval_getitem_loads_full_audio_and_annotations_csv(self):
        dataset = RealDESEDDataset(
            root=str(self.root),
            split="validation",
            sample_rate=self.sample_rate,
            frame_hz=self.frame_hz,
            classes=TEST_CLASSES,
        )
        item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["audio"].shape, (1, int(self.sample_rate * 1.2)))
        self.assertEqual(item["labels"].shape, (2, 12))
        self.assertEqual(item["labels"][0].sum().item(), 2.0)
        self.assertEqual(item["labels"][1].sum().item(), 0.0)
        self.assertEqual(item["timestamps"].shape, (12,))
        self.assertAlmostEqual(item["duration"], 1.2, places=4)

    def test_load_audio_resamples_and_mixes_stereo_to_mono(self):
        dataset = self._train_dataset()
        source_rate = self.sample_rate // 2
        stereo = torch.stack(
            [
                torch.ones(source_rate, dtype=torch.float32),
                torch.zeros(source_rate, dtype=torch.float32),
            ]
        )

        def fake_resample(audio, orig_freq, new_freq):
            self.assertEqual(orig_freq, source_rate)
            self.assertEqual(new_freq, self.sample_rate)
            return torch.nn.functional.interpolate(
                audio.unsqueeze(0),
                size=self.sample_rate,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        with mock.patch("dataset.dataset.torchaudio.load", return_value=(stereo, source_rate)):
            with mock.patch.object(dataset_module.torchaudio_functional, "resample", side_effect=fake_resample):
                audio, duration = dataset._load_audio("train_a.wav")

        self.assertEqual(audio.shape, (1, self.sample_rate))
        self.assertTrue(torch.allclose(audio, torch.full((1, self.sample_rate), 0.5)))
        self.assertAlmostEqual(duration, 1.0)

    def test_collate_pads_train_and_eval_batches(self):
        train_item = self._train_dataset("Uniform Soft")[0]
        short_train = dict(train_item)
        short_train["audio"] = train_item["audio"][:, : self.sample_rate // 2]
        short_train["soft_labels"] = train_item["soft_labels"][:, :5]
        short_train["hard_labels"] = train_item["hard_labels"][:, :5]
        short_train["timestamps"] = train_item["timestamps"][:5]
        short_train["duration"] = 0.5

        train_batch = collate_fn([short_train, train_item])
        self.assertEqual(train_batch["audio"].shape, (2, 1, self.sample_rate))
        self.assertEqual(train_batch["soft_labels"].shape, (2, 2, 10))
        self.assertEqual(train_batch["hard_labels"].shape, (2, 2, 10))
        self.assertEqual(train_batch["timestamps"][0, -1].item(), -1.0)

        eval_item = RealDESEDDataset(
            root=str(self.root),
            split="validation",
            sample_rate=self.sample_rate,
            frame_hz=self.frame_hz,
            classes=TEST_CLASSES,
        )[0]
        short_eval = dict(eval_item)
        short_eval["audio"] = eval_item["audio"][:, : self.sample_rate // 2]
        short_eval["labels"] = eval_item["labels"][:, :5]
        short_eval["timestamps"] = eval_item["timestamps"][:5]
        short_eval["duration"] = 0.5

        eval_batch = collate_fn([short_eval, eval_item])
        self.assertEqual(eval_batch["audio"].shape, (2, 1, int(self.sample_rate * 1.2)))
        self.assertEqual(eval_batch["labels"].shape, (2, 2, 12))
        self.assertEqual(eval_batch["timestamps"][0, -1].item(), -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
