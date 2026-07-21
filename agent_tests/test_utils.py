import types
import unittest
from pathlib import Path
import sys
from unittest import mock

import torch

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if __name__ == "__main__":
        print(f"Skipping utils tests: missing dependency {exc.name!r}.")
        raise SystemExit(0)
    raise

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["bell", "coffee"]


def _install_sed_scores_eval_stub():
    sed_scores_eval = types.ModuleType("sed_scores_eval")
    sed_scores_eval.intersection_based = types.SimpleNamespace(
        psds=lambda *args, **kwargs: (0.0, {class_name: 0.0 for class_name in CLASS_NAMES})
    )
    sed_scores_eval.segment_based = types.SimpleNamespace(
        auroc=lambda *args, **kwargs: ({"mean": 0.0}, {})
    )
    sys.modules["sed_scores_eval"] = sed_scores_eval


try:
    import sed_scores_eval  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name in {"sed_scores_eval", "pkg_resources"}:
        _install_sed_scores_eval_stub()
    elif __name__ == "__main__":
        print(f"Skipping utils tests: missing dependency {exc.name!r}.")
        raise SystemExit(0)
    else:
        raise

try:
    import utils.evaluation as evaluation_module
    from utils.augment import RandomResizeCrop, apply_mixup_spectrogram
    from utils.evaluation import (
        add_psds_logs,
        apply_median_filter,
        build_metric_logs,
        can_compute_psds_metrics,
        categorize_recording_device,
        events_to_frame_preds,
        events_to_tuples,
        normalize_metadata_value,
        normalize_recording_device_name,
        preds_to_score_df,
        recording_environment_class_names,
        recording_environment_labels_for_metadata,
        store_metadata,
        subset_by_attribute,
        subset_by_label_membership,
    )
    from utils.inference import sliding_window_inference
except ModuleNotFoundError as exc:
    if __name__ == "__main__":
        print(f"Skipping utils tests: missing dependency {exc.name!r}.")
        raise SystemExit(0)
    raise


class UtilsTest(unittest.TestCase):
    def test_preds_to_score_df_uses_frame_grid_and_class_columns(self):
        preds = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
        timestamps = torch.tensor([0.05, 0.15, 0.25])

        score_df = preds_to_score_df(preds, timestamps, CLASS_NAMES, frame_hz=10)

        self.assertEqual(list(score_df.columns), ["onset", "offset", "bell", "coffee"])
        self.assertEqual(score_df["onset"].tolist(), [0.0, 0.1, 0.2])
        self.assertEqual(score_df["offset"].tolist(), [0.1, 0.2, 0.3])
        self.assertTrue(torch.allclose(torch.tensor(score_df["bell"].tolist()), torch.tensor([0.1, 0.2, 0.3])))
        self.assertTrue(torch.allclose(torch.tensor(score_df["coffee"].tolist()), torch.tensor([0.9, 0.8, 0.7])))

    def test_events_to_frame_preds_supports_three_and_four_tuple_events(self):
        timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
        events = [
            (0.1, 0.3, "bell"),
            (0.0, 0.2, "coffee", 0.99),
            (0.0, 1.0, "ignored"),
        ]

        frame_preds = events_to_frame_preds(events, CLASS_NAMES, timestamps)

        self.assertTrue(torch.equal(frame_preds[0], torch.tensor([0, 1, 1, 0, 0]).float()))
        self.assertTrue(torch.equal(frame_preds[1], torch.tensor([1, 1, 0, 0, 0]).float()))

    def test_events_to_tuples_filters_empty_or_negative_duration_events(self):
        events = [
            {"start": "0.1", "end": "0.5", "label": "bell"},
            {"start": "0.5", "end": "0.5", "label": "coffee"},
            {"start": "0.8", "end": "0.7", "label": "coffee"},
        ]

        self.assertEqual(events_to_tuples(events), [(0.1, 0.5, "bell")])

    def test_metadata_normalization_and_device_categorization(self):
        self.assertEqual(normalize_metadata_value(None), "")
        self.assertEqual(normalize_metadata_value("  Static "), "Static")
        self.assertEqual(normalize_recording_device_name("Google_Pixel-8  Pro"), "google pixel 8 pro")

        self.assertEqual(categorize_recording_device("iPhone 15 Pro"), "ios")
        self.assertEqual(categorize_recording_device("Samsung Galaxy S24"), "android")
        self.assertEqual(categorize_recording_device("Field Recorder"), "other")
        self.assertEqual(categorize_recording_device(""), "other")

    def test_environment_helpers_and_store_metadata_route_by_stage(self):
        class DummyDataset:
            metadata = [
                {"recording_environment": "kitchen; hallway"},
                {"recording_environment": "kitchen; bedroom"},
            ]

            @staticmethod
            def _split_list(value):
                return [item.strip() for item in value.split(";") if item.strip()]

        environments = recording_environment_class_names(DummyDataset())
        self.assertEqual(environments, ["bedroom", "hallway", "kitchen"])
        self.assertEqual(
            recording_environment_labels_for_metadata(
                {"recording_environment": ["kitchen", "garage"]},
                environments,
            ),
            ["kitchen"],
        )

        val_device_placements = {}
        test_device_placements = {}
        val_environment_labels = {}
        test_environment_labels = {}
        val_device_categories = {}
        test_device_categories = {}

        store_metadata(
            "val",
            "file_a",
            {
                "device_placement": " Static ",
                "recording_environment": ["kitchen", "garage"],
                "recording_device": "iPad Air",
            },
            environments,
            val_device_placements,
            test_device_placements,
            val_environment_labels,
            test_environment_labels,
            val_device_categories,
            test_device_categories,
        )
        store_metadata(
            "test",
            "file_b",
            {
                "device_placement": "",
                "recording_environment": ["bedroom"],
                "recording_device": "Pixel 8",
            },
            environments,
            val_device_placements,
            test_device_placements,
            val_environment_labels,
            test_environment_labels,
            val_device_categories,
            test_device_categories,
        )

        self.assertEqual(val_device_placements, {"file_a": "static"})
        self.assertEqual(test_device_placements, {})
        self.assertEqual(val_environment_labels, {"file_a": ["kitchen"]})
        self.assertEqual(test_environment_labels, {"file_b": ["bedroom"]})
        self.assertEqual(val_device_categories, {"file_a": "ios"})
        self.assertEqual(test_device_categories, {"file_b": "android"})

    def test_subset_helpers_and_psds_log_format(self):
        predictions = {"a": "pred-a", "b": "pred-b", "c": "pred-c"}
        ground_truth = {"a": [(0, 1, "bell")], "b": [(0, 1, "coffee")], "c": []}
        durations = {"a": 1.0, "b": 2.0, "c": 3.0}

        self.assertTrue(can_compute_psds_metrics(ground_truth, CLASS_NAMES))
        self.assertFalse(can_compute_psds_metrics({"a": ground_truth["a"]}, CLASS_NAMES))

        attr_subset = subset_by_attribute(
            predictions,
            ground_truth,
            durations,
            {"a": "static", "b": "mobile", "missing": "static"},
            "static",
        )
        self.assertEqual(attr_subset, ({"a": "pred-a"}, {"a": ground_truth["a"]}, {"a": 1.0}))
        self.assertIsNone(subset_by_attribute(predictions, ground_truth, durations, {}, "static"))

        label_subset = subset_by_label_membership(
            predictions,
            ground_truth,
            durations,
            {"a": ["kitchen"], "b": ["bedroom"], "c": ["kitchen"]},
            "kitchen",
        )
        self.assertEqual(
            label_subset,
            (
                {"a": "pred-a", "c": "pred-c"},
                {"a": ground_truth["a"], "c": []},
                {"a": 1.0, "c": 3.0},
            ),
        )
        self.assertIsNone(subset_by_label_membership(predictions, ground_truth, durations, {}, "kitchen"))

        logs = {}
        add_psds_logs(
            logs,
            "prefix",
            ((0.2, {"bell": 0.1, "coffee": 0.3}), (0.4, {"bell": 0.2, "coffee": 0.6})),
        )
        self.assertEqual(
            logs,
            {
                "prefix_psds1": 0.2,
                "prefix_psds1_macro": 0.2,
                "prefix_psds2": 0.4,
                "prefix_psds2_macro": 0.4,
            },
        )

    def test_median_filter_returns_filtered_copies(self):
        score_df = pd.DataFrame(
            {
                "onset": [0.0, 0.1, 0.2],
                "offset": [0.1, 0.2, 0.3],
                "bell": [0.0, 1.0, 0.0],
                "coffee": [1.0, 0.0, 1.0],
            }
        )

        filtered = apply_median_filter({"file_a": score_df}, CLASS_NAMES, median_window=3)

        self.assertIsNot(filtered["file_a"], score_df)
        self.assertEqual(filtered["file_a"]["onset"].tolist(), [0.0, 0.1, 0.2])
        self.assertEqual(filtered["file_a"]["bell"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(filtered["file_a"]["coffee"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(score_df["bell"].tolist(), [0.0, 1.0, 0.0])

    def test_build_metric_logs_includes_core_and_grouped_metrics(self):
        score_df = pd.DataFrame(
            {
                "onset": [0.0, 0.5],
                "offset": [0.5, 1.0],
                "bell": [0.8, 0.1],
                "coffee": [0.2, 0.9],
            }
        )
        predictions = {"a": score_df, "b": score_df.copy()}
        ground_truth = {
            "a": [(0.0, 0.5, "bell"), (0.5, 1.0, "coffee")],
            "b": [(0.0, 0.5, "bell"), (0.5, 1.0, "coffee")],
        }
        durations = {"a": 1.0, "b": 1.0}

        fake_metrics = ((0.5, {"bell": 0.4, "coffee": 0.6}), (0.7, {"bell": 0.8, "coffee": 0.2}))
        with mock.patch.object(evaluation_module, "compute_psds_metrics", return_value=fake_metrics):
            with mock.patch.object(evaluation_module.sed_scores_eval.segment_based, "auroc", return_value=({"mean": 0.9}, {})):
                logs, psds1_per_class, psds2_per_class = build_metric_logs(
                    predictions,
                    ground_truth,
                    durations,
                    {"a": "static", "b": "mobile"},
                    {"a": ["kitchen"], "b": ["bedroom"]},
                    {"a": "ios", "b": "android"},
                    "val",
                    CLASS_NAMES,
                    ["kitchen", "bedroom"],
                )

        self.assertEqual(logs["val/psds1"], 0.5)
        self.assertEqual(logs["val/psds1_macro"], 0.5)
        self.assertEqual(logs["val/psds2"], 0.7)
        self.assertEqual(logs["val/pauroc"], 0.9)
        self.assertEqual(logs["val_classwise/psds1/bell"], 0.4)
        self.assertEqual(logs["val_classwise/psds2/coffee"], 0.2)
        self.assertIn("val_placement/static_psds1", logs)
        self.assertIn("val_device/ios_psds1", logs)
        self.assertIn("val_recording_env/kitchen_psds1", logs)
        self.assertEqual(psds1_per_class, fake_metrics[0][1])
        self.assertEqual(psds2_per_class, fake_metrics[1][1])

    def test_sliding_window_inference_average_and_max_stitching(self):
        audio = torch.zeros(1, 6)

        class SequentialPredictor:
            def __init__(self):
                self.next_value = 1.0

            def __call__(self, chunk_batch):
                outputs = []
                for _ in range(chunk_batch.shape[0]):
                    outputs.append(torch.full((1, 4), self.next_value))
                    self.next_value += 2.0
                return {"event_logits": torch.stack(outputs)}

        average = sliding_window_inference(
            audio,
            SequentialPredictor(),
            num_classes=1,
            chunk_size=2.0,
            hop_size=1.0,
            sample_rate=2,
            frame_hz=2,
            inference_batch_size=2,
            stitching="average",
            triangular_filter_floor=1.0,
        )
        self.assertTrue(torch.allclose(average, torch.tensor([[1.0, 1.0, 2.0, 2.0, 4.0, 4.0]])))

        max_stitched = sliding_window_inference(
            audio.unsqueeze(0),
            SequentialPredictor(),
            num_classes=1,
            chunk_size=2.0,
            hop_size=1.0,
            sample_rate=2,
            frame_hz=2,
            inference_batch_size=2,
            stitching="max",
        )
        self.assertTrue(torch.equal(max_stitched, torch.tensor([[1.0, 1.0, 3.0, 3.0, 5.0, 5.0]])))

        with self.assertRaises(ValueError):
            sliding_window_inference(
                audio,
                SequentialPredictor(),
                num_classes=1,
                chunk_size=2.0,
                hop_size=1.0,
                sample_rate=2,
                frame_hz=2,
                inference_batch_size=2,
                stitching="bad-mode",
            )

    def test_mixup_can_skip_or_mix_with_patched_randomness(self):
        mel = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
        labels = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])

        with mock.patch("utils.augment.torch.rand", return_value=torch.tensor([1.0])):
            skipped_mel, skipped_labels = apply_mixup_spectrogram(mel, labels, mixup_p=0.0, mixup_alpha=0.2)
        self.assertIs(skipped_mel, mel)
        self.assertIs(skipped_labels, labels)

        with mock.patch("utils.augment.torch.rand", return_value=torch.tensor([0.0])):
            with mock.patch("utils.augment.torch.randperm", return_value=torch.tensor([1, 0])):
                with mock.patch("utils.augment.torch.distributions.Beta") as beta:
                    beta.return_value.sample.return_value = torch.tensor(0.25)
                    mixed_mel, mixed_labels = apply_mixup_spectrogram(mel, labels, mixup_p=1.0, mixup_alpha=0.2)

        expected_mel = 0.75 * mel + 0.25 * mel[torch.tensor([1, 0])]
        expected_labels = 0.75 * labels + 0.25 * labels[torch.tensor([1, 0])]
        self.assertTrue(torch.equal(mixed_mel, expected_mel))
        self.assertTrue(torch.equal(mixed_labels, expected_labels))

    def test_random_resize_crop_params_and_forward_shape(self):
        with mock.patch("utils.augment.np.random.uniform", side_effect=[1.0, 1.0]):
            with mock.patch("utils.augment.random.randint", side_effect=[1, 2]):
                self.assertEqual(
                    RandomResizeCrop.get_params(
                        virtual_crop_size=(8, 10),
                        in_size=(4, 5),
                        time_scale=(1.0, 1.0),
                        freq_scale=(1.0, 1.0),
                    ),
                    (1, 2, 4, 5),
                )

        lms = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
        crop = RandomResizeCrop(
            virtual_crop_scale=(1.0, 1.0),
            freq_scale=(1.0, 1.0),
            time_scale=(1.0, 1.0),
        )
        output = crop(lms)
        self.assertEqual(output.shape, lms.shape)
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
