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


def tune_csebbs_predictor(predictions, ground_truth, durations, output_dir):
    try:
        from sebbs import csebbs
    except ImportError as exc:
        raise ImportError(
            "cSEBBS evaluation requires the 'sebbs' package. "
            "Install it to run the default test evaluation."
        ) from exc

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
            selection_fn=csebbs.select_best_psds,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0.0,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return predictor, best_psds_values


def apply_csebbs(predictions, csebbs_predictor, class_names):
    if csebbs_predictor is None:
        raise ValueError("cSEBBS predictor is not fitted.")

    scores_dict = {
        file_id: score_df.sort_values("onset")
        for file_id, score_df in predictions.items()
    }
    events_dict = csebbs_predictor.predict(scores_dict)

    csebbs_predictions = {}
    for file_id, score_df in scores_dict.items():
        timestamps = torch.as_tensor(score_df["onset"].values)
        frame_preds = events_to_frame_preds(
            events_dict[file_id],
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
