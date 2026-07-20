import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["bell", "coffee"]


def _install_import_stubs():
    lightning = types.ModuleType("pytorch_lightning")

    class FakeLightningModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logged = []
            self.logged_dicts = []

        def log(self, *args, **kwargs):
            self.logged.append((args, kwargs))

        def log_dict(self, *args, **kwargs):
            self.logged_dicts.append((args, kwargs))

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
            instance = cls(*args, **kwargs)
            instance.loaded_checkpoint_path = checkpoint_path
            return instance

    class FakeTrainer:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_calls = []
            self.test_calls = []
            self.estimated_stepping_batches = 12
            self.optimizers = [types.SimpleNamespace(param_groups=[{"lr": 0.001}])]
            FakeTrainer.instances.append(self)

        def fit(self, model, train_dl, val_dl):
            self.fit_calls.append((model, train_dl, val_dl))

        def test(self, model, dataloaders=None):
            self.test_calls.append((model, dataloaders))

    lightning.LightningModule = FakeLightningModule
    lightning.Trainer = FakeTrainer

    callbacks = types.ModuleType("pytorch_lightning.callbacks")

    class FakeModelCheckpoint:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.best_model_path = os.path.join(kwargs["dirpath"], "best.ckpt")
            FakeModelCheckpoint.instances.append(self)

    class FakeEarlyStopping:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeEarlyStopping.instances.append(self)

    callbacks.ModelCheckpoint = FakeModelCheckpoint
    callbacks.EarlyStopping = FakeEarlyStopping

    loggers = types.ModuleType("pytorch_lightning.loggers")

    class FakeWandbLogger:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeWandbLogger.instances.append(self)

    loggers.WandbLogger = FakeWandbLogger

    transformers = types.ModuleType("transformers")
    transformers.scheduler_calls = []

    def fake_scheduler(optimizer, num_warmup_steps, num_training_steps):
        call = {
            "optimizer": optimizer,
            "num_warmup_steps": num_warmup_steps,
            "num_training_steps": num_training_steps,
        }
        transformers.scheduler_calls.append(call)
        return call

    transformers.get_cosine_schedule_with_warmup = fake_scheduler

    wandb = types.ModuleType("wandb")
    wandb.finish_calls = 0

    def finish():
        wandb.finish_calls += 1

    wandb.finish = finish

    atst_module = types.ModuleType("models.atstframe.ATSTF_wrapper")

    class FakeATSTWrapper(torch.nn.Module):
        embed_dim = 4

    atst_module.ATSTWrapper = FakeATSTWrapper

    prediction_module = types.ModuleType("models.prediction_wrapper")

    class FakePredictionsWrapper(torch.nn.Module):
        def __init__(self, backbone, checkpoint, embed_dim, seq_len, seq_model_type, head_type):
            super().__init__()
            self.backbone = backbone
            self.checkpoint = checkpoint
            self.embed_dim = embed_dim
            self.seq_len = seq_len
            self.num_features = 3

        def mel_forward(self, audio):
            batch = audio.shape[0]
            return torch.zeros(batch, 1, 2, self.seq_len, device=audio.device)

        def forward(self, mel):
            batch = mel.shape[0]
            return torch.ones(batch, self.seq_len, self.num_features, device=mel.device)

    prediction_module.PredictionsWrapper = FakePredictionsWrapper

    evaluation = types.ModuleType("utils.evaluation")
    evaluation.apply_median_filter = lambda predictions, class_names, median_window: predictions
    evaluation.build_metric_logs = lambda *args, **kwargs: ({}, {}, {})
    evaluation.events_to_tuples = lambda events: [(event["start"], event["end"], event["label"]) for event in events]
    evaluation.preds_to_score_df = lambda *args, **kwargs: None
    evaluation.recording_environment_class_names = lambda *datasets: ["kitchen", "hallway"]
    evaluation.store_metadata = lambda *args, **kwargs: None

    csebbs = types.ModuleType("utils.csebbs")
    csebbs.apply_csebbs = lambda predictions, predictor, class_names: predictions
    csebbs.tune_csebbs_predictor = lambda predictions, ground_truth, durations, output_dir, class_names=None: ("predictor", {})

    inference = types.ModuleType("utils.inference")
    inference.sliding_window_inference = lambda **kwargs: torch.zeros(kwargs["num_classes"], 2)

    sys.modules["pytorch_lightning"] = lightning
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.loggers"] = loggers
    sys.modules["transformers"] = transformers
    sys.modules["wandb"] = wandb
    sys.modules["models.atstframe.ATSTF_wrapper"] = atst_module
    sys.modules["models.prediction_wrapper"] = prediction_module
    sys.modules["utils.evaluation"] = evaluation
    sys.modules["utils.csebbs"] = csebbs
    sys.modules["utils.inference"] = inference


_install_import_stubs()
train_module = importlib.import_module("train")


class FakeDataset:
    calls = []

    def __init__(
        self,
        root,
        split,
        sample_rate,
        chunk_size=None,
        frame_hz=25,
        classes=None,
        aggregation="Majority",
        annotator_weight_alpha=16.0,
        include_reviewed_train_files=False,
    ):
        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.frame_hz = frame_hz
        self.classes = list(classes) if classes is not None else ["bell", "coffee"]
        self.aggregation = aggregation
        self.annotator_weight_alpha = annotator_weight_alpha
        self.include_reviewed_train_files = include_reviewed_train_files
        self.metadata = [{"recording_environment": "kitchen"}]
        FakeDataset.calls.append(self)

    @staticmethod
    def _split_list(value):
        return [item.strip() for item in value.split(";") if item.strip()]

    def __len__(self):
        return 2


class FakeDataLoader:
    instances = []

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        FakeDataLoader.instances.append(self)

    def __iter__(self):
        return iter(())


class FakeTrainingPLModule:
    instances = []
    loaded_instances = []

    def __init__(
        self,
        config,
        class_names,
        recording_environment_class_names,
        pretrained_checkpoint,
        frame_hz,
        use_csebbs,
    ):
        self.config = config
        self.class_names = list(class_names)
        self.recording_environment_class_names = list(recording_environment_class_names)
        self.pretrained_checkpoint = pretrained_checkpoint
        self.frame_hz = frame_hz
        self.use_csebbs = use_csebbs
        self.csebbs_tuning_dataloader = None
        FakeTrainingPLModule.instances.append(self)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        instance = cls(**kwargs)
        instance.loaded_checkpoint_path = checkpoint_path
        cls.loaded_instances.append(instance)
        return instance


def make_config(output_dir):
    return types.SimpleNamespace(
        dataset_path="data",
        output_dir=output_dir,
        experiment_name="unit-train",
        wandb_project="RealDESED-tests",
        sample_rate=16000,
        chunk_size=10.0,
        hop_size=5.0,
        inference_batch_size=4,
        triangular_filter_floor=0.3,
        sliding_window_stitching="average",
        train_annotation_aggregation="Weighted Soft",
        annotator_alpha=2.0,
        include_reviewed_train_files=True,
        batch_size=8,
        num_workers=0,
        num_devices=1,
        precision=32,
        check_val_every_n_epoch=2,
        early_stopping_patience=3,
        accumulate_grad_batches=4,
        n_epochs=5,
        max_lr=0.01,
        weight_decay=0.02,
        warmup_steps=7,
        mixup_p=0.0,
        mixup_alpha=0.2,
        freq_warp_p=0.0,
        median_window=5,
    )


class TrainPyTest(unittest.TestCase):
    def setUp(self):
        FakeDataset.calls.clear()
        FakeDataLoader.instances.clear()
        FakeTrainingPLModule.instances.clear()
        FakeTrainingPLModule.loaded_instances.clear()
        train_module.pl.Trainer.instances.clear()
        train_module.ModelCheckpoint.instances.clear()
        train_module.EarlyStopping.instances.clear()
        train_module.WandbLogger.instances.clear()
        train_module.wandb.finish_calls = 0

    def test_plmodule_training_label_selection(self):
        config = types.SimpleNamespace(
            chunk_size=1.0,
            train_annotation_aggregation="Uniform Soft",
            max_lr=0.01,
            weight_decay=0.1,
            warmup_steps=3,
        )
        module = train_module.PLModule(
            config,
            class_names=CLASS_NAMES,
            recording_environment_class_names=["kitchen"],
            frame_hz=4,
            use_csebbs=False,
        )
        batch = {
            "soft_labels": torch.ones(1, 2, 4),
            "hard_labels": torch.zeros(1, 2, 4),
        }

        self.assertIs(module._training_labels(batch), batch["soft_labels"])
        module.config.train_annotation_aggregation = "Majority"
        self.assertIs(module._training_labels(batch), batch["hard_labels"])

    def test_plmodule_configure_optimizers_uses_config_and_trainer_steps(self):
        config = types.SimpleNamespace(
            chunk_size=1.0,
            train_annotation_aggregation="Majority",
            max_lr=0.01,
            weight_decay=0.1,
            warmup_steps=3,
        )
        module = train_module.PLModule(
            config,
            class_names=CLASS_NAMES,
            recording_environment_class_names=["kitchen"],
            frame_hz=4,
            use_csebbs=False,
        )
        module.trainer = types.SimpleNamespace(estimated_stepping_batches=123)

        optimizers, schedulers = module.configure_optimizers()

        self.assertEqual(optimizers[0].param_groups[0]["lr"], 0.01)
        self.assertEqual(optimizers[0].param_groups[0]["weight_decay"], 0.1)
        self.assertEqual(schedulers[0]["interval"], "step")
        self.assertEqual(schedulers[0]["scheduler"]["num_warmup_steps"], 3)
        self.assertEqual(schedulers[0]["scheduler"]["num_training_steps"], 123)

    def test_train_wires_datasets_dataloaders_callbacks_trainer_and_test(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = make_config(output_dir)

            with mock.patch.object(train_module, "RealDESEDDataset", FakeDataset):
                with mock.patch.object(train_module, "DataLoader", FakeDataLoader):
                    with mock.patch.object(train_module, "PLModule", FakeTrainingPLModule):
                        train_module.train(
                            config,
                            model_name="UnitModel",
                            pretrained_checkpoint="unit.ckpt",
                            pretrained="unit-pretrained",
                            frame_hz=50,
                            use_csebbs=True,
                        )

        self.assertEqual(config.model_name, "UnitModel")
        self.assertEqual(config.pretrained, "unit-pretrained")
        self.assertEqual(config.frame_hz, 50)
        self.assertTrue(config.csebbs_apply)

        self.assertEqual([dataset.split for dataset in FakeDataset.calls], ["train", "validation", "test"])
        train_ds, val_ds, test_ds = FakeDataset.calls
        self.assertEqual(train_ds.root, "data")
        self.assertEqual(train_ds.chunk_size, 10.0)
        self.assertEqual(train_ds.frame_hz, 50)
        self.assertEqual(train_ds.aggregation, "Weighted Soft")
        self.assertEqual(train_ds.annotator_weight_alpha, 2.0)
        self.assertTrue(train_ds.include_reviewed_train_files)
        self.assertEqual(val_ds.classes, train_ds.classes)
        self.assertEqual(test_ds.classes, train_ds.classes)

        self.assertEqual(len(FakeDataLoader.instances), 3)
        train_dl, val_dl, test_dl = FakeDataLoader.instances
        self.assertEqual(train_dl.kwargs["batch_size"], 8)
        self.assertTrue(train_dl.kwargs["shuffle"])
        self.assertTrue(train_dl.kwargs["drop_last"])
        self.assertFalse(val_dl.kwargs["shuffle"])
        self.assertFalse(test_dl.kwargs["shuffle"])
        self.assertIs(train_dl.kwargs["collate_fn"], train_module.collate_fn)
        self.assertIs(val_dl.kwargs["collate_fn"], train_module.collate_fn)
        self.assertIs(test_dl.kwargs["collate_fn"], train_module.collate_fn)

        logger = train_module.WandbLogger.instances[0]
        self.assertEqual(logger.kwargs["project"], "RealDESED-tests")
        self.assertEqual(logger.kwargs["name"], "unit-train")
        self.assertTrue(logger.kwargs["save_dir"].endswith(os.path.join("experiment_dumps", "wandb")))

        checkpoint = train_module.ModelCheckpoint.instances[0]
        self.assertEqual(checkpoint.kwargs["monitor"], "val/psds1_macro")
        self.assertEqual(checkpoint.kwargs["mode"], "max")
        self.assertEqual(checkpoint.kwargs["filename"], "best-{epoch}")

        early_stop = train_module.EarlyStopping.instances[0]
        self.assertEqual(early_stop.kwargs["monitor"], "val/psds1_macro")
        self.assertEqual(early_stop.kwargs["patience"], 3)

        trainer = train_module.pl.Trainer.instances[0]
        self.assertEqual(trainer.kwargs["max_epochs"], 5)
        self.assertEqual(trainer.kwargs["devices"], 1)
        self.assertEqual(trainer.kwargs["precision"], 32)
        self.assertEqual(trainer.kwargs["check_val_every_n_epoch"], 2)
        self.assertEqual(trainer.kwargs["accumulate_grad_batches"], 4)
        self.assertEqual(len(trainer.kwargs["callbacks"]), 2)

        self.assertEqual(trainer.fit_calls, [(FakeTrainingPLModule.instances[0], train_dl, val_dl)])
        self.assertEqual(len(FakeTrainingPLModule.loaded_instances), 1)
        best_model = FakeTrainingPLModule.loaded_instances[0]
        self.assertEqual(best_model.loaded_checkpoint_path, checkpoint.best_model_path)
        self.assertIs(best_model.csebbs_tuning_dataloader, val_dl)
        self.assertEqual(trainer.test_calls, [(best_model, test_dl)])
        self.assertEqual(train_module.wandb.finish_calls, 1)

    def test_train_does_not_set_csebbs_tuning_dataloader_when_disabled(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = make_config(output_dir)

            with mock.patch.object(train_module, "RealDESEDDataset", FakeDataset):
                with mock.patch.object(train_module, "DataLoader", FakeDataLoader):
                    with mock.patch.object(train_module, "PLModule", FakeTrainingPLModule):
                        train_module.train(config, use_csebbs=False)

        best_model = FakeTrainingPLModule.loaded_instances[0]
        self.assertFalse(config.csebbs_apply)
        self.assertIsNone(best_model.csebbs_tuning_dataloader)
        self.assertFalse(best_model.use_csebbs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
