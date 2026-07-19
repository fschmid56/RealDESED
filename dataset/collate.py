from typing import Dict, List

import torch


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_audio_len = max(item["audio"].shape[1] for item in batch)
    max_label_len = max(
        item.get("labels", item.get("hard_labels")).shape[1]
        for item in batch
    )

    def pad_audio(audio: torch.Tensor) -> torch.Tensor:
        pad = max_audio_len - audio.shape[1]
        if pad > 0:
            audio = torch.nn.functional.pad(audio, (0, pad))
        return audio

    def pad_labels(labels: torch.Tensor) -> torch.Tensor:
        pad = max_label_len - labels.shape[1]
        if pad > 0:
            labels = torch.nn.functional.pad(labels, (0, pad))
        return labels

    def pad_timestamps(timestamps: torch.Tensor) -> torch.Tensor:
        pad = max_label_len - timestamps.shape[0]
        if pad > 0:
            timestamps = torch.nn.functional.pad(timestamps, (0, pad), value=-1.0)
        return timestamps

    output = {
        "audio": torch.stack([pad_audio(item["audio"]) for item in batch]),
        "filename": [item["filename"] for item in batch],
        "timestamps": torch.stack([pad_timestamps(item["timestamps"]) for item in batch]),
        "duration": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
        "events": [item["events"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }

    if "soft_labels" in batch[0]:
        output["soft_labels"] = torch.stack([pad_labels(item["soft_labels"]) for item in batch])
        output["hard_labels"] = torch.stack([pad_labels(item["hard_labels"]) for item in batch])
    else:
        output["labels"] = torch.stack([pad_labels(item["labels"]) for item in batch])

    return output
