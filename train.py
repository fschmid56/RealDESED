import argparse
import os
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset.collate import collate_fn
from dataset.dataset import RealDESEDDataset
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from utils.augment import RandomResizeCrop, apply_mixup_spectrogram
from utils.csebbs import apply_csebbs, tune_csebbs_predictor
from utils.evaluation import (
    apply_median_filter,
    build_metric_logs,
    events_to_tuples,
    preds_to_score_df,
    recording_environment_class_names,
    store_metadata,
)
from utils.inference import sliding_window_inference


class PLModule(pl.LightningModule):
    def __init__(
        self,
        config,
        class_names,
        recording_environment_class_names,
        pretrained_checkpoint="ATST-F_strong_1",
        frame_hz=25,
        use_csebbs=True,
    ):
        super().__init__()
        self.config = config
        self.class_names = list(class_names)
        self.recording_environment_class_names = list(recording_environment_class_names)
        self.frame_hz = frame_hz
        self.use_csebbs = use_csebbs

        backbone = ATSTWrapper()

        self.model = PredictionsWrapper(
            backbone,
            checkpoint=pretrained_checkpoint,
            embed_dim=getattr(backbone, "embed_dim", 768),
            seq_len=int(config.chunk_size * self.frame_hz),
            seq_model_type=None,
            head_type=None,
        )
        self.event_head = nn.Linear(self.model.num_features, len(self.class_names))

        self.bce_logits = nn.BCEWithLogitsLoss()
        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.val_predictions = {}
        self.val_ground_truth = {}
        self.val_durations = {}
        self.val_device_placements = {}
        self.val_recording_environment_labels = {}
        self.val_recording_device_categories = {}
        self.test_predictions = {}
        self.test_ground_truth = {}
        self.test_durations = {}
        self.test_device_placements = {}
        self.test_recording_environment_labels = {}
        self.test_recording_device_categories = {}
        self.csebbs_predictor = None
        self.csebbs_tuning_dataloader = None

    def forward(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        mel = self.model.mel_forward(audio)
        return self.forward_from_mel(mel)

    def forward_from_mel(self, mel):
        features = self.model(mel)
        logits = self.event_head(features).transpose(1, 2)
        return {"event_logits": logits}

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        labels = self._training_labels(batch)

        if audio.dim() == 3:
            audio = audio.squeeze(1)

        mel = self.model.mel_forward(audio)
        mel, labels = apply_mixup_spectrogram(
            mel,
            labels,
            self.config.mixup_p,
            self.config.mixup_alpha,
        )

        if self.config.freq_warp_p > random.random():
            mel = self.freq_warp(mel.squeeze(1)).unsqueeze(1)

        outputs = self.forward_from_mel(mel)
        loss = self.bce_logits(outputs["event_logits"], labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True)

        return loss

    def _training_labels(self, batch):
        if self.config.train_annotation_aggregation in {
            "Uniform Soft",
            "Weighted Soft",
        }:
            return batch["soft_labels"]
        return batch["hard_labels"]

    def _store_metadata(self, stage, file_id, metadata):
        store_metadata(
            stage,
            file_id,
            metadata,
            self.recording_environment_class_names,
            self.val_device_placements,
            self.test_device_placements,
            self.val_recording_environment_labels,
            self.test_recording_environment_labels,
            self.val_recording_device_categories,
            self.test_recording_device_categories,
        )

    def sliding_window_inference(self, audio):
        return sliding_window_inference(
            audio=audio,
            predict_fn=self,
            num_classes=len(self.class_names),
            chunk_size=self.config.chunk_size,
            hop_size=self.config.hop_size,
            sample_rate=self.config.sample_rate,
            frame_hz=self.frame_hz,
            inference_batch_size=self.config.inference_batch_size,
            stitching=getattr(self.config, "sliding_window_stitching", "average"),
            triangular_filter_floor=self.config.triangular_filter_floor,
        )

    def _shared_eval_step(self, batch, stage):
        audios = batch["audio"]
        labels = batch["labels"]
        timestamps = batch["timestamps"]
        filenames = batch["filename"]
        durations = batch["duration"]
        events = batch["events"]
        metadata_batch = batch["metadata"]

        losses = []
        predictions = self.val_predictions if stage == "val" else self.test_predictions
        ground_truth = self.val_ground_truth if stage == "val" else self.test_ground_truth
        duration_store = self.val_durations if stage == "val" else self.test_durations

        for i in range(audios.shape[0]):
            ts = timestamps[i]
            valid_len = (ts >= 0).sum().item()
            ts = ts[:valid_len]
            label = labels[i, :, :valid_len]

            frame_duration = 1.0 / self.frame_hz
            audio_len = int((ts[-1].item() + 0.5 * frame_duration) * self.config.sample_rate)
            audio = audios[i, :, :audio_len]

            logits = self.sliding_window_inference(audio)
            min_len = min(logits.shape[1], label.shape[1])
            logits = logits[:, :min_len]
            label = label[:, :min_len]
            ts_aligned = ts[:min_len]

            losses.append(self.bce_logits(logits.unsqueeze(0), label.unsqueeze(0)))
            probs = torch.sigmoid(logits)

            score_df = preds_to_score_df(
                probs,
                ts_aligned,
                self.class_names,
                frame_hz=self.frame_hz,
            )
            score_df = score_df.sort_index()
            score_df = score_df[~score_df.index.duplicated(keep="first")]

            file_id = filenames[i].replace(".wav", "")
            predictions[file_id] = score_df.copy()
            ground_truth[file_id] = events_to_tuples(events[i])
            duration_store[file_id] = durations[i].item() if isinstance(durations, torch.Tensor) else durations[i]
            self._store_metadata(stage, file_id, metadata_batch[i])

        loss = torch.stack(losses).mean()
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=audios.shape[0],
        )

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")

    def _use_csebbs_for_stage(self, stage):
        return stage == "test" and self.use_csebbs

    @torch.no_grad()
    def _collect_raw_predictions(self, dataloader):
        predictions = {}
        ground_truth = {}
        durations = {}
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        for batch in dataloader:
            audios = batch["audio"]
            labels = batch["labels"]
            timestamps = batch["timestamps"]
            filenames = batch["filename"]
            durations_batch = batch["duration"]
            batch_events = batch["events"]

            for i in range(audios.shape[0]):
                ts = timestamps[i]
                valid_len = (ts >= 0).sum().item()
                label = labels[i, :, :valid_len]
                ts = ts[:valid_len]

                frame_duration = 1.0 / self.frame_hz
                audio_len = int((ts[-1].item() + 0.5 * frame_duration) * self.config.sample_rate)
                audio = audios[i, :, :audio_len].to(device)

                logits = self.sliding_window_inference(audio)
                min_len = min(logits.shape[1], label.shape[1])
                probs = torch.sigmoid(logits[:, :min_len])
                ts_aligned = ts[:min_len].to(probs.device)

                score_df = preds_to_score_df(
                    probs,
                    ts_aligned,
                    self.class_names,
                    frame_hz=self.frame_hz,
                )
                score_df = score_df.sort_index()
                score_df = score_df[~score_df.index.duplicated(keep="first")]

                file_id = filenames[i].replace(".wav", "")
                predictions[file_id] = score_df.copy()
                ground_truth[file_id] = events_to_tuples(batch_events[i])
                durations[file_id] = (
                    durations_batch[i].item()
                    if isinstance(durations_batch, torch.Tensor)
                    else durations_batch[i]
                )

        if was_training:
            self.train()

        return predictions, ground_truth, durations

    def _fit_csebbs_predictor(self):
        if self.csebbs_tuning_dataloader is None:
            raise ValueError("csebbs_tuning_dataloader must be set before testing with cSEBBS.")

        print("\nCollecting validation predictions for cSEBBS tuning...")
        predictions, ground_truth, durations = self._collect_raw_predictions(self.csebbs_tuning_dataloader)

        print("Tuning cSEBBS on validation predictions...")
        self.csebbs_predictor, best_psds_values = tune_csebbs_predictor(
            predictions,
            ground_truth,
            durations,
            self.config.output_dir,
            self.class_names,
        )
        print(f"Best cSEBBS validation PSDS values: {best_psds_values}")

    def _log_epoch_metrics(self, stage):
        predictions = self.val_predictions if stage == "val" else self.test_predictions
        ground_truth = self.val_ground_truth if stage == "val" else self.test_ground_truth
        durations = self.val_durations if stage == "val" else self.test_durations
        device_placements = self.val_device_placements if stage == "val" else self.test_device_placements
        environment_labels = (
            self.val_recording_environment_labels
            if stage == "val"
            else self.test_recording_environment_labels
        )
        device_categories = (
            self.val_recording_device_categories
            if stage == "val"
            else self.test_recording_device_categories
        )

        if len(predictions) == 0:
            return

        metric_sets = [(stage, apply_median_filter(predictions, self.class_names, self.config.median_window))]
        if self._use_csebbs_for_stage(stage):
            metric_sets = [
                ("test_median", metric_sets[0][1]),
                ("test_csebbs", apply_csebbs(predictions, self.csebbs_predictor, self.class_names)),
            ]

        logs = {}
        psds1_per_class = {}
        psds2_per_class = {}
        for prefix, metric_predictions in metric_sets:
            metric_logs, psds1_per_class, psds2_per_class = build_metric_logs(
                metric_predictions,
                ground_truth,
                durations,
                device_placements,
                environment_labels,
                device_categories,
                prefix,
                self.class_names,
                self.recording_environment_class_names,
            )
            logs.update(metric_logs)

        self.log_dict(logs, prog_bar=True)

        sorted_psds = sorted(psds1_per_class.items(), key=lambda item: item[1])
        print(f"\n===== {stage.upper()} PSDS1 per class (worst to best) =====")
        for class_name, score in sorted_psds:
            print(f"{class_name:28s}: {score:.4f}")
        print("=================================================\n")

        sorted_psds = sorted(psds2_per_class.items(), key=lambda item: item[1])
        print(f"\n===== {stage.upper()} PSDS2 per class (worst to best) =====")
        for class_name, score in sorted_psds:
            print(f"{class_name:28s}: {score:.4f}")
        print("=================================================\n")

        predictions.clear()
        ground_truth.clear()
        durations.clear()
        device_placements.clear()
        environment_labels.clear()
        device_categories.clear()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_start(self):
        if self._use_csebbs_for_stage("test") and self.csebbs_predictor is None:
            self._fit_csebbs_predictor()

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(
    config,
    model_name="ATST-F",
    pretrained_checkpoint="ATST-F_strong_1",
    pretrained="strong",
    frame_hz=25,
    use_csebbs=True,
):
    config.model_name = model_name
    config.pretrained = pretrained
    config.frame_hz = frame_hz
    config.csebbs_apply = use_csebbs

    experiment_dump_dir = os.path.join(config.output_dir, "experiment_dumps")
    wandb_dir = os.path.join(experiment_dump_dir, "wandb")
    wandb_cache_dir = os.path.join(experiment_dump_dir, "wandb_cache")
    wandb_config_dir = os.path.join(experiment_dump_dir, "wandb_config")
    lightning_root_dir = os.path.join(experiment_dump_dir, "lightning_logs")
    checkpoint_dir = os.path.join(experiment_dump_dir, "checkpoints")

    for path in (experiment_dump_dir, wandb_dir, wandb_cache_dir, wandb_config_dir, lightning_root_dir, checkpoint_dir):
        os.makedirs(path, exist_ok=True)

    os.environ["WANDB_DIR"] = wandb_dir
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
    os.environ["WANDB_CONFIG_DIR"] = wandb_config_dir

    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.experiment_name,
        config=vars(config),
        save_dir=wandb_dir,
    )

    train_ds = RealDESEDDataset(
        root=config.dataset_path,
        split="train",
        sample_rate=config.sample_rate,
        chunk_size=config.chunk_size,
        frame_hz=frame_hz,
        aggregation=config.train_annotation_aggregation,
        annotator_weight_alpha=config.annotator_alpha,
        include_reviewed_train_files=config.include_reviewed_train_files,
    )
    val_ds = RealDESEDDataset(
        root=config.dataset_path,
        split="validation",
        sample_rate=config.sample_rate,
        frame_hz=frame_hz,
        classes=train_ds.classes,
    )
    test_ds = RealDESEDDataset(
        root=config.dataset_path,
        split="test",
        sample_rate=config.sample_rate,
        frame_hz=frame_hz,
        classes=train_ds.classes,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    environment_class_names = recording_environment_class_names(train_ds, val_ds, test_ds)

    model = PLModule(
        config,
        train_ds.classes,
        environment_class_names,
        pretrained_checkpoint=pretrained_checkpoint,
        frame_hz=frame_hz,
        use_csebbs=use_csebbs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/psds1_macro",
        mode="max",
        save_top_k=1,
        save_last=False,
        filename="best-{epoch}",
        auto_insert_metric_name=False,
        dirpath=checkpoint_dir,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/psds1_macro",
        mode="max",
        patience=config.early_stopping_patience,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        default_root_dir=lightning_root_dir,
        accelerator="auto",
        devices=config.num_devices,
        precision=config.precision,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    trainer.fit(model, train_dl, val_dl)

    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"\nLoading best checkpoint from:\n{best_ckpt_path}\n")

    best_model = PLModule.load_from_checkpoint(
        best_ckpt_path,
        config=config,
        class_names=train_ds.classes,
        recording_environment_class_names=environment_class_names,
        pretrained_checkpoint=pretrained_checkpoint,
        frame_hz=frame_hz,
        use_csebbs=use_csebbs,
    )
    if use_csebbs:
        best_model.csebbs_tuning_dataloader = val_dl
    trainer.test(best_model, dataloaders=test_dl)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--experiment_name", type=str, default="RealDESED")
    parser.add_argument("--wandb_project", type=str, default="RealDESED")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=999)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    parser.add_argument("--chunk_size", type=float, default=10.0)
    parser.add_argument("--hop_size", type=float, default=5.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--inference_batch_size", type=int, default=64)
    parser.add_argument(
        "--triangular_filter_floor",
        type=float,
        default=0.3,
        help="Minimum edge weight for triangular sliding-window inference weights. Must be in [0, 1].",
    )
    parser.add_argument(
        "--sliding_window_stitching",
        type=str,
        choices=["average", "max"],
        default="average",
        help="How overlapping sliding-window predictions are stitched together.",
    )

    parser.add_argument(
        "--train_annotation_aggregation",
        type=str,
        choices=sorted(RealDESEDDataset.AGGREGATIONS),
        default="Majority",
        help="How train annotations_raw.csv is aggregated for training labels.",
    )
    parser.add_argument(
        "--annotator_alpha",
        type=float,
        default=16.0,
        help=(
            "Exponent for Weighted Soft annotator scores. "
            "0 uses uniform weights, 1 keeps current weights, >1 sharpens differences."
        ),
    )
    parser.add_argument(
        "--include_reviewed_train_files",
        action="store_true",
        help="Use accepted reviewed rows from annotations_raw.csv for training files where they exist.",
    )
    parser.add_argument("--mixup_p", type=float, default=0.5)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--freq_warp_p", type=float, default=0.5)
    parser.add_argument("--median_window", type=int, default=9)

    args = parser.parse_args()
    if not 0.0 <= args.triangular_filter_floor <= 1.0:
        parser.error("--triangular_filter_floor must be in [0, 1]")
    train(args)
