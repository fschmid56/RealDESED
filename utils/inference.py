import torch
import torch.nn.functional as F


def sliding_window_inference(
    audio,
    predict_fn,
    num_classes,
    chunk_size,
    hop_size,
    sample_rate,
    frame_hz,
    inference_batch_size,
    stitching="average",
    triangular_filter_floor=0.3,
):
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    assert audio.dim() == 2

    window_size = int(chunk_size * sample_rate)
    hop_size = int(hop_size * sample_rate)
    total_len = audio.shape[1]
    n_frames_total = int(total_len / sample_rate * frame_hz)

    if stitching == "average":
        logits_sum = torch.zeros(num_classes, n_frames_total, device=audio.device)
        counts = torch.zeros(n_frames_total, device=audio.device)
    elif stitching == "max":
        logits_max = torch.full(
            (num_classes, n_frames_total),
            -torch.inf,
            device=audio.device,
        )
    else:
        raise ValueError(f"Unknown sliding-window stitching mode: {stitching}")

    chunks = []
    starts = []
    for start in range(0, total_len, hop_size):
        chunk = audio[:, start:start + window_size]
        if chunk.shape[1] < window_size:
            chunk = F.pad(chunk, (0, window_size - chunk.shape[1]))
        chunks.append(chunk)
        starts.append(start)

    chunks = torch.stack(chunks)

    for i in range(0, len(chunks), inference_batch_size):
        chunk_batch = chunks[i:i + inference_batch_size]
        outputs = predict_fn(chunk_batch)
        logits_batch = outputs["event_logits"]

        for j in range(logits_batch.shape[0]):
            chunk_logits = logits_batch[j]
            start_sample = starts[i + j]
            start_frame = int(start_sample * frame_hz / sample_rate)
            end_frame = min(start_frame + chunk_logits.shape[1], n_frames_total)
            valid_len = end_frame - start_frame

            if stitching == "average":
                t = torch.linspace(0, 1, chunk_logits.shape[1], device=audio.device)
                weights = triangular_filter_floor + (1.0 - triangular_filter_floor) * (
                    1.0 - torch.abs(2 * t - 1)
                )
                weights = weights[:valid_len]

                logits_sum[:, start_frame:end_frame] += chunk_logits[:, :valid_len] * weights.unsqueeze(0)
                counts[start_frame:end_frame] += weights
            else:
                logits_max[:, start_frame:end_frame] = torch.maximum(
                    logits_max[:, start_frame:end_frame],
                    chunk_logits[:, :valid_len],
                )

    if stitching == "average":
        return logits_sum / counts.clamp_min(1e-6).unsqueeze(0)

    return logits_max
