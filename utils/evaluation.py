import re

import numpy as np
import pandas as pd
import sed_scores_eval
import torch


def preds_to_score_df(preds, timestamps, class_names, frame_hz=25):
    """
    Convert frame probabilities [classes, frames] to the sed_scores_eval format.
    """
    probs = preds.cpu().numpy().T
    frame_duration = 1.0 / frame_hz
    start = timestamps.cpu().numpy()[0] - 0.5 * frame_duration

    onset = start + np.arange(probs.shape[0]) * frame_duration
    offset = onset + frame_duration

    df = pd.DataFrame(probs, columns=class_names)
    df.insert(0, "offset", np.round(offset, decimals=4))
    df.insert(0, "onset", np.round(onset, decimals=4))
    return df


def events_to_frame_preds(events, class_names, timestamps, device=None):
    """
    Convert event intervals to binary frame predictions on the given timestamp grid.
    """
    frame_preds = torch.zeros(len(class_names), len(timestamps), device=device)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    for event in events:
        if len(event) == 4:
            onset, offset, event_label, _ = event
        else:
            onset, offset, event_label = event

        if event_label not in class_to_idx:
            continue

        class_idx = class_to_idx[event_label]
        mask = (timestamps >= float(onset)) & (timestamps < float(offset))
        frame_preds[class_idx, mask] = 1.0

    return frame_preds


def events_to_tuples(events):
    return [
        (float(event["start"]), float(event["end"]), event["label"])
        for event in events
        if float(event["end"]) > float(event["start"])
    ]


def recording_environment_class_names(*datasets):
    environments = set()
    for dataset in datasets:
        for row in dataset.metadata:
            environments.update(dataset._split_list(row.get("recording_environment", "")))
    return sorted(environments)


def normalize_metadata_value(value):
    return str(value).strip() if value is not None else ""


def normalize_recording_device_name(value):
    normalized = normalize_metadata_value(value).lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def categorize_recording_device(value):
    normalized = normalize_recording_device_name(value)

    if not normalized:
        return "other"

    ios_patterns = [
        r"\biphone\b",
        r"\biphone\s*\d",
        r"\biphone\d",
        r"\biphone\s*[a-z]",
        r"\biphon\b",
        r"\biphon\s*\d",
        r"\bipad\b",
        r"\bipad\s*[a-z0-9]",
        r"\bi pad\b",
    ]
    if any(re.search(pattern, normalized) for pattern in ios_patterns):
        return "ios"

    android_patterns = [
        r"\bandroid phone\b",
        r"\bandroid\b",
        r"\bsamsung\b",
        r"\bgalaxy\b",
        r"\bpixel\b",
        r"\bgoogle pixel\b",
        r"\bxiaomi\b",
        r"\bxioami\b",
        r"\bredmi\b",
        r"\bpoco\b",
        r"\bmotorola\b",
        r"\bmoto\b",
        r"\bhonor\b",
        r"\bhuawei\b",
        r"\boneplus\b",
        r"\bonepluss\b",
        r"\boppo\b",
        r"\bvivo\b",
        r"\brealme\b",
        r"\bnothing phone\b",
        r"\bcmf phone\b",
        r"\bfairphone\b",
        r"\bzenfone\b",
        r"\bhmd skyline\b",
        r"\bcat s\d",
    ]
    if any(re.search(pattern, normalized) for pattern in android_patterns):
        return "android"

    return "other"


def recording_environment_labels_for_metadata(metadata, environment_class_names):
    return [
        environment
        for environment in metadata.get("recording_environment", [])
        if environment in environment_class_names
    ]


def store_metadata(
    stage,
    file_id,
    metadata,
    environment_class_names,
    val_device_placements,
    test_device_placements,
    val_recording_environment_labels,
    test_recording_environment_labels,
    val_recording_device_categories,
    test_recording_device_categories,
):
    device_placements = val_device_placements if stage == "val" else test_device_placements
    environment_labels = (
        val_recording_environment_labels
        if stage == "val"
        else test_recording_environment_labels
    )
    device_categories = (
        val_recording_device_categories
        if stage == "val"
        else test_recording_device_categories
    )

    placement = normalize_metadata_value(metadata.get("device_placement")).lower()
    if placement:
        device_placements[file_id] = placement

    environment_labels[file_id] = recording_environment_labels_for_metadata(
        metadata,
        environment_class_names,
    )
    device_categories[file_id] = categorize_recording_device(metadata.get("recording_device"))


def compute_psds_metrics(predictions, ground_truth, durations):
    psds1 = sed_scores_eval.intersection_based.psds(
        predictions,
        ground_truth,
        durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0,
        alpha_st=1,
        num_jobs=1,
    )
    psds2 = sed_scores_eval.intersection_based.psds(
        predictions,
        ground_truth,
        durations,
        dtc_threshold=0.1,
        gtc_threshold=0.1,
        cttc_threshold=None,
        alpha_ct=0,
        alpha_st=1,
        num_jobs=1,
    )
    return (psds1[0], psds1[1]), (psds2[0], psds2[1])


def can_compute_psds_metrics(ground_truth, class_names):
    classes_with_ground_truth = {
        label
        for events in ground_truth.values()
        for _, _, label in events
    }
    return set(class_names).issubset(classes_with_ground_truth)


def subset_by_attribute(predictions, ground_truth, durations, attribute_values, attribute_value):
    subset_keys = [
        file_id for file_id, value in attribute_values.items()
        if value == attribute_value and file_id in predictions
    ]

    if len(subset_keys) == 0:
        return None

    return (
        {file_id: predictions[file_id] for file_id in subset_keys},
        {file_id: ground_truth[file_id] for file_id in subset_keys},
        {file_id: durations[file_id] for file_id in subset_keys},
    )


def subset_by_label_membership(predictions, ground_truth, durations, label_lists, label):
    subset_keys = [
        file_id for file_id, labels in label_lists.items()
        if label in labels and file_id in predictions
    ]

    if len(subset_keys) == 0:
        return None

    return (
        {file_id: predictions[file_id] for file_id in subset_keys},
        {file_id: ground_truth[file_id] for file_id in subset_keys},
        {file_id: durations[file_id] for file_id in subset_keys},
    )


def add_psds_logs(logs, prefix, psds_metrics):
    (psds1_value, psds1_per_class), (psds2_value, psds2_per_class) = psds_metrics
    logs[f"{prefix}_psds1"] = psds1_value
    logs[f"{prefix}_psds1_macro"] = np.mean(list(psds1_per_class.values()))
    logs[f"{prefix}_psds2"] = psds2_value
    logs[f"{prefix}_psds2_macro"] = np.mean(list(psds2_per_class.values()))


def apply_median_filter(predictions, class_names, median_window):
    filtered = {}
    for file_id, score_df in predictions.items():
        output_df = score_df.copy()
        output_df[class_names] = output_df[class_names].rolling(
            window=median_window,
            center=True,
        ).median().bfill().ffill()
        filtered[file_id] = output_df
    return filtered


def build_metric_logs(
    predictions,
    ground_truth,
    durations,
    device_placements,
    environment_labels,
    device_categories,
    prefix,
    class_names,
    environment_class_names,
):
    (psds1_value, psds1_per_class), (psds2_value, psds2_per_class) = compute_psds_metrics(
        predictions,
        ground_truth,
        durations,
    )
    pauroc = sed_scores_eval.segment_based.auroc(
        predictions,
        ground_truth,
        durations,
        max_fpr=0.1,
        segment_length=1.0,
        num_jobs=1,
    )

    logs = {
        f"{prefix}/psds1": psds1_value,
        f"{prefix}/psds1_macro": np.mean(list(psds1_per_class.values())),
        f"{prefix}/psds2": psds2_value,
        f"{prefix}/psds2_macro": np.mean(list(psds2_per_class.values())),
        f"{prefix}/pauroc": pauroc[0]["mean"],
    }
    for class_name, score in psds1_per_class.items():
        logs[f"{prefix}_classwise/psds1/{class_name}"] = score
    for class_name, score in psds2_per_class.items():
        logs[f"{prefix}_classwise/psds2/{class_name}"] = score

    for placement in ("static", "mobile"):
        placement_subset = subset_by_attribute(
            predictions,
            ground_truth,
            durations,
            device_placements,
            placement,
        )
        if placement_subset is None or not can_compute_psds_metrics(placement_subset[1], class_names):
            continue
        add_psds_logs(
            logs,
            f"{prefix}_placement/{placement}",
            compute_psds_metrics(*placement_subset),
        )

    for category in ("ios", "android", "other"):
        category_subset = subset_by_attribute(
            predictions,
            ground_truth,
            durations,
            device_categories,
            category,
        )
        if category_subset is None or not can_compute_psds_metrics(category_subset[1], class_names):
            continue
        add_psds_logs(
            logs,
            f"{prefix}_device/{category}",
            compute_psds_metrics(*category_subset),
        )

    for environment in environment_class_names:
        environment_subset = subset_by_label_membership(
            predictions,
            ground_truth,
            durations,
            environment_labels,
            environment,
        )
        if environment_subset is None or not can_compute_psds_metrics(environment_subset[1], class_names):
            continue
        add_psds_logs(
            logs,
            f"{prefix}_recording_env/{environment}",
            compute_psds_metrics(*environment_subset),
        )

    return logs, psds1_per_class, psds2_per_class
