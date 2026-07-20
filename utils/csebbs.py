import os
import shutil
import tempfile

import pandas as pd
import torch

from utils.evaluation import events_to_frame_preds


def write_csebbs_inputs(predictions, ground_truth, durations, base_tmp_dir):
    scores_dir = os.path.join(base_tmp_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    for file_id, score_df in predictions.items():
        score_df.to_csv(
            os.path.join(scores_dir, f"file_{file_id}.tsv"),
            sep="\t",
            index=False,
        )

    ground_truth_rows = []
    for file_id, events in ground_truth.items():
        for start, end, label in events:
            ground_truth_rows.append(
                {
                    "filename": f"file_{file_id}.wav",
                    "onset": start,
                    "offset": end,
                    "event_label": label,
                }
            )
    pd.DataFrame(ground_truth_rows).to_csv(
        os.path.join(base_tmp_dir, "ground_truth.tsv"),
        sep="\t",
        index=False,
    )

    duration_rows = [
        {"filename": f"file_{file_id}.wav", "duration": duration}
        for file_id, duration in durations.items()
    ]
    pd.DataFrame(duration_rows).to_csv(
        os.path.join(base_tmp_dir, "durations.tsv"),
        sep="\t",
        index=False,
    )

    return (
        scores_dir,
        os.path.join(base_tmp_dir, "ground_truth.tsv"),
        os.path.join(base_tmp_dir, "durations.tsv"),
    )


def _prediction_class_names(predictions):
    class_names = []
    for score_df in predictions.values():
        for column in [
            column
            for column in score_df.columns
            if column not in {"onset", "offset"}
        ]:
            if column not in class_names:
                class_names.append(column)
    return class_names


def _ground_truth_class_names(ground_truth):
    return {
        label
        for events in ground_truth.values()
        for _, _, label in events
    }


def _tuning_class_names(predictions, ground_truth, class_names):
    prediction_classes = set(_prediction_class_names(predictions))
    ground_truth_classes = _ground_truth_class_names(ground_truth)
    return [
        class_name
        for class_name in class_names
        if class_name in prediction_classes and class_name in ground_truth_classes
    ]


def _predictor_class_names(csebbs_predictor, class_names):
    predictor_class_names = getattr(csebbs_predictor, "sound_classes", None)
    if predictor_class_names is None:
        return list(class_names)

    return [
        class_name
        for class_name in predictor_class_names
        if class_name in class_names
    ]


def _class_sebbs(sebbs, class_name):
    return {
        audio_id: [
            sebb
            for sebb in sebbs_for_audio
            if len(sebb) >= 3 and sebb[2] == class_name
        ]
        for audio_id, sebbs_for_audio in sebbs.items()
    }


def _single_class_psds_value(
    intersection_based,
    sed_scores_from_sebbs,
    sebbs,
    ground_truth,
    audio_durations,
    class_name,
    psds_kwargs,
):
    class_sebbs = _class_sebbs(sebbs, class_name)
    if not any(class_sebbs.values()):
        return 0.0

    scores = sed_scores_from_sebbs(
        class_sebbs,
        sound_classes=[class_name],
        audio_duration=audio_durations,
    )
    try:
        return intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            **psds_kwargs,
        )[1].get(class_name, 0.0)
    except IndexError as exc:
        if "out of bounds" not in str(exc):
            raise
        return 0.0


def _select_best_psds_robust(
    csebbs_module,
    predictors,
    ground_truth,
    audio_durations,
    sound_classes,
    audio_ids=None,
    dtc_threshold=0.7,
    gtc_threshold=0.7,
    cttc_threshold=None,
    alpha_ct=0.0,
    unit_of_time="hour",
    max_efpr=100.0,
    classwise=True,
    num_jobs=1,
    **kwargs,
):
    if audio_ids is not None:
        audio_ids = list(audio_ids)
        predictors = [
            (predictor, {audio_id: sebbs[audio_id] for audio_id in audio_ids})
            for predictor, sebbs in predictors
        ]
        ground_truth = {audio_id: ground_truth[audio_id] for audio_id in audio_ids}
        audio_durations = {audio_id: audio_durations[audio_id] for audio_id in audio_ids}

    psds_kwargs = {
        "dtc_threshold": dtc_threshold,
        "gtc_threshold": gtc_threshold,
        "cttc_threshold": cttc_threshold,
        "alpha_ct": alpha_ct,
        "unit_of_time": unit_of_time,
        "max_efpr": max_efpr,
        "num_jobs": num_jobs,
    }

    from sed_scores_eval import intersection_based
    from sebbs.utils import sed_scores_from_sebbs

    best_step_filter_length = {}
    best_merge_threshold_abs = {}
    best_merge_threshold_rel = {}
    best_values = {}

    for predictor, sebbs in predictors:
        single_class_psds = {
            class_name: _single_class_psds_value(
                intersection_based,
                sed_scores_from_sebbs,
                sebbs,
                ground_truth,
                audio_durations,
                class_name,
                psds_kwargs,
            )
            for class_name in sound_classes
        }
        mean = sum(single_class_psds.values()) / len(single_class_psds)
        for class_name, class_psds in single_class_psds.items():
            value = class_psds if classwise else mean
            if class_name not in best_values or value > best_values[class_name]:
                best_values[class_name] = value
                best_step_filter_length[class_name] = predictor.step_filter_length
                best_merge_threshold_abs[class_name] = predictor.merge_threshold_abs
                best_merge_threshold_rel[class_name] = predictor.merge_threshold_rel

    csebbs_predictor = csebbs_module.CSEBBsPredictor(
        step_filter_length=best_step_filter_length,
        merge_threshold_rel=best_merge_threshold_rel,
        merge_threshold_abs=best_merge_threshold_abs,
        sound_classes=list(sound_classes),
    )
    return csebbs_predictor, best_values


def tune_csebbs_predictor(predictions, ground_truth, durations, output_dir, class_names=None):
    try:
        from sebbs import csebbs
    except ImportError as exc:
        raise ImportError(
            "cSEBBS evaluation requires the 'sebbs' package. "
            "Install it to run the default test evaluation."
        ) from exc

    class_names = list(class_names) if class_names is not None else _prediction_class_names(predictions)
    if len(class_names) == 0:
        raise ValueError("Cannot tune cSEBBS without prediction class columns.")

    tuning_class_names = _tuning_class_names(predictions, ground_truth, class_names)
    if len(tuning_class_names) == 0:
        raise ValueError("Cannot tune cSEBBS without classes present in both predictions and ground truth.")
    skipped_class_names = [
        class_name
        for class_name in class_names
        if class_name not in tuning_class_names
    ]
    if skipped_class_names:
        print(
            "Skipping cSEBBS tuning for classes without validation ground truth: "
            f"{skipped_class_names}"
        )

    def select_best_psds_with_classes(predictors, *args, **kwargs):
        selected_predictor, best_psds_values = _select_best_psds_robust(
            csebbs,
            predictors,
            *args,
            sound_classes=tuning_class_names,
            **kwargs,
        )
        return selected_predictor, best_psds_values

    tmp_root = os.path.join(output_dir, "experiment_dumps")
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="csebbs_tune_", dir=tmp_root)

    try:
        scores, ground_truth_path, durations_path = write_csebbs_inputs(
            predictions,
            ground_truth,
            durations,
            tmp_dir,
        )
        predictor, best_psds_values = csebbs.tune(
            scores=scores,
            ground_truth=ground_truth_path,
            audio_durations=durations_path,
            step_filter_lengths=(.32, .48, .64),
            merge_thresholds_abs=(.15, .2, .3),
            merge_thresholds_rel=(1.5, 2., 3.),
            selection_fn=select_best_psds_with_classes,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0.0,
            alpha_st=1.0,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return predictor, best_psds_values


def apply_csebbs(predictions, csebbs_predictor, class_names):
    if csebbs_predictor is None:
        raise ValueError("cSEBBS predictor is not fitted.")

    predictor_class_names = _predictor_class_names(csebbs_predictor, class_names)
    if len(predictor_class_names) == 0:
        raise ValueError("cSEBBS predictor has no classes matching the evaluation classes.")

    scores_dict = {
        file_id: score_df.sort_values("onset")[["onset", "offset", *predictor_class_names]]
        for file_id, score_df in predictions.items()
    }
    events_dict = csebbs_predictor.predict(scores_dict)

    csebbs_predictions = {}
    for file_id, score_df in scores_dict.items():
        timestamps = torch.as_tensor(score_df["onset"].values)
        frame_preds = events_to_frame_preds(
            events_dict.get(file_id, []),
            class_names,
            timestamps,
            device=None,
        )

        output_df = pd.DataFrame(
            {
                "onset": score_df["onset"].values,
                "offset": score_df["offset"].values,
            }
        )
        for class_idx, class_name in enumerate(class_names):
            output_df[class_name] = frame_preds[class_idx].cpu().numpy()

        csebbs_predictions[file_id] = output_df

    return csebbs_predictions
