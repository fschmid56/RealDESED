# RealDESED: A Real-World Benchmark Dataset for Domestic Sound Event Detection

This repository contains the official baseline training and evaluation code for the RealDESED domestic sound event detection benchmark.

Dataset: [Zenodo](https://zenodo.org/records/20056072) | Paper: arXiv coming soon; currently under review for the DCASE Workshop 2026

## Dataset at a Glance

- 🏠 **5,710** real-world recordings
- 👥 **652** participants collecting recordings in their own homes
- 🎧 **645** annotators providing temporal event labels
- ⏱️ **37.85 hours** of audio
- 🔊 **15** domestic sound event classes
- 📝 **64,430** temporally annotated sound events
- 📱 Recordings from a wide range of consumer devices
- 🚶 Static and mobile recording setups
- 🏡 Rich metadata including recording environment, device information, scene descriptions, and more
- ✅ Reviewed validation and test annotations for reliable benchmarking

RealDESED is a new benchmark for domestic sound event detection (SED) consisting exclusively of real-world recordings collected in participants' homes.

Unlike existing SED benchmarks that rely heavily on synthetic soundscapes or web-crawled audio, RealDESED captures realistic recording conditions, including different devices, environments, recording setups, background sounds, and naturally occurring event co-occurrences.

Besides the dataset itself, we provide a strong transformer-based baseline together with extensive analyses of annotation aggregation, long-form inference, post-processing, and recording metadata.

## Quick Setup

First, install Conda by following the [Anaconda installation guide](https://www.anaconda.com/docs/getting-started/main).

Clone this repository and run the setup script:

```bash
git clone https://github.com/fschmid56/RealDESED.git
cd RealDESED
bash setup.sh
```

The setup script creates a Conda environment named `realdesed`, installs the Python dependencies, downloads the public RealDESED dataset from Zenodo into `data/downloads/`, and extracts the splits to `data/train/`, `data/validation/`, and `data/test/`.

Activate the environment:

```bash
conda activate realdesed
```

To reproduce the weighted annotator labels used by `Weighted Soft` ($\alpha$=16), compute the annotator quality scores:

```bash
python compute_annotator_score.py
```

By default this reads `data/train/annotations_raw.csv` and writes
`data/annotator_quality/annotator_scores_train.json`. To use a different
dataset location, pass `--root /path/to/dataset`.

To run the ATST-F baseline with weighted soft labels:

This uses the reviewed training annotations where available and corresponds to row `#10`, `Weighted + Reviewed`, in Table 1 of the arXiv paper.

Weights & Biases run: coming soon

```bash
python train.py \
  --dataset_path data \
  --output_dir runs \
  --experiment_name atst_f_weighted_soft \
  --wandb_project RealDESED \
  --train_annotation_aggregation "Weighted Soft" \
  --annotator_alpha 16 \
  --include_reviewed_train_files
```

To reproduce the majority-voting results:

Weights & Biases run: coming soon

```bash
python train.py \
  --dataset_path data \
  --output_dir runs \
  --experiment_name atst_f_majority \
  --wandb_project RealDESED \
  --train_annotation_aggregation Majority
```

Hardware requirements for reproducing the baseline experiments are modest: an NVIDIA GeForce RTX 2080 Ti is sufficient to run the baseline in less than two hours.

## Weights & Biases Logs

Training logs are written to Weights & Biases under the project passed with `--wandb_project` and the run name passed with `--experiment_name`.

The most important logged metrics are:

- `train/loss`: training loss.
- `val/loss`: validation loss.
- `val/psds1_macro` and `val/psds2_macro`: validation PSDS macro scores used to monitor model quality; the best checkpoint is selected by `val/psds1_macro`.
- `val/pauroc`: validation partial AUROC.
- `val_classwise/psds1/*` and `val_classwise/psds2/*`: class-wise validation PSDS scores.
- `test/*`: test-set metrics after applying the global median filter controlled by `--median_window`.
- `test_classwise/*`: class-wise test metrics.

Grouped PSDS metrics are also logged for recording metadata subsets such as device placement, device category, and recording environment.

## Dataset Layout

The training script expects the extracted dataset to use this layout:

```text
data/
  train/
    audio/
    metadata.csv
    annotations_raw.csv
  validation/
    audio/
    metadata.csv
    annotations.csv
  test/
    audio/
    metadata.csv
    annotations.csv
```

The public baseline trains on the `train` split, selects the best checkpoint on
validation PSDS1 macro, then evaluates the best checkpoint on the test split.

## Command Line Arguments

The main training entry point is `train.py`. The most commonly changed arguments are:

- `--dataset_path`: path to the extracted RealDESED dataset, defaulting to `data`.
- `--output_dir` and `--experiment_name`: where local outputs are written and how the run is named.
- `--wandb_project`: Weights & Biases project used for logging.
- `--batch_size`, `--num_workers`, `--num_devices`, and `--precision`: hardware and dataloader settings.
- `--n_epochs`, `--max_lr`, `--weight_decay`, `--warmup_steps`, and `--accumulate_grad_batches`: optimization settings.
- `--train_annotation_aggregation`, `--annotator_alpha`, and `--include_reviewed_train_files`: training-label construction settings.
- `--chunk_size`, `--hop_size`, `--inference_batch_size`, and `--sliding_window_stitching`: long-form inference settings.
- `--median_window`: median-filter post-processing window.

Run `python train.py --help` for the complete list of available options.

## Annotation Aggregation

Table 1 in the paper compares different training-label aggregation strategies. To reproduce it, run the ATST-F baseline three times for each row and change the annotation-related command line arguments as listed below.

Use this base command and replace the final aggregation arguments for each row:

```bash
python train.py \
  --dataset_path data \
  --output_dir runs \
  --wandb_project RealDESED \
  --experiment_name <run_name>
```

Rows `#1` to `#8` use only the unreviewed multi-annotator training labels:

| Row | Method | Command line arguments |
| --- | --- | --- |
| `#1` | `Random (Fixed)` | `--train_annotation_aggregation "Random (Fixed)"` |
| `#2` | `Random (Epoch)` | `--train_annotation_aggregation "Random (Epoch)"` |
| `#3` | `Majority` | `--train_annotation_aggregation Majority` |
| `#4` | `Intersection` | `--train_annotation_aggregation Intersection` |
| `#5` | `Union` | `--train_annotation_aggregation Union` |
| `#6` | `Collector` | `--train_annotation_aggregation Collector` |
| `#7` | `Uniform Soft` | `--train_annotation_aggregation "Uniform Soft"` |
| `#8` | `Weighted Soft` ($\alpha$=16) | `--train_annotation_aggregation "Weighted Soft" --annotator_alpha 16` |

Rows `#9` and `#10` additionally use reviewed training annotations where available:

| Row | Method | Command line arguments |
| --- | --- | --- |
| `#9` | `Majority + Reviewed` | `--train_annotation_aggregation Majority --include_reviewed_train_files` |
| `#10` | `Weighted + Reviewed` | `--train_annotation_aggregation "Weighted Soft" --annotator_alpha 16 --include_reviewed_train_files` |

Before running rows `#8` and `#10`, generate the annotator quality scores once with:

```bash
python compute_annotator_score.py
```

## Citations

If you use the RealDESED dataset or this code, please cite the RealDESED paper, which is about to appear on arXiv and is currently under review for the DCASE Workshop 2026.

If you use the AudioSet Strong pre-trained baseline, please also cite:

```bibtex
@inproceedings{DBLP:conf/icassp/SchmidMFSPW25,
  author       = {Florian Schmid and
                  Tobias Morocutti and
                  Francesco Foscarin and
                  Jan Schl{\"{u}}ter and
                  Paul Primus and
                  Gerhard Widmer},
  title        = {Effective Pre-Training of Audio Transformers for Sound Event Detection},
  booktitle    = {2025 {IEEE} International Conference on Acoustics, Speech and Signal
                  Processing, {ICASSP} 2025, Hyderabad, India, April 6-11, 2025},
  pages        = {1--5},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/ICASSP49660.2025.10888942},
  doi          = {10.1109/ICASSP49660.2025.10888942}
}
```

The ATST-Frame architecture is published in:

```bibtex
@article{DBLP:journals/taslp/LiSL24,
  author       = {Xian Li and
                  Nian Shao and
                  Xiaofei Li},
  title        = {Self-Supervised Audio Teacher-Student Transformer for Both Clip-Level
                  and Frame-Level Tasks},
  journal      = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
  volume       = {32},
  pages        = {1336--1351},
  year         = {2024},
  url          = {https://doi.org/10.1109/TASLP.2024.3352248},
  doi          = {10.1109/TASLP.2024.3352248}
}
```

## License

This codebase is released under the [MIT License](LICENSE).

The RealDESED dataset is hosted on [Zenodo](https://zenodo.org/records/20056072). The dataset licenses specified on Zenodo apply: audio recordings and their corresponding metadata follow the per-file terms listed in `metadata.csv`, which are either CC0 or CC BY, while the remaining metadata and annotations are licensed under CC BY 4.0.
