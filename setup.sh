#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-realdesed}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
DATA_DIR="${DATA_DIR:-data}"
DOWNLOAD_DIR="$DATA_DIR/downloads"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required but was not found on PATH." >&2
  exit 1
fi

download_file() {
  local url="$1"
  local output="$2"

  if [[ -s "$output" ]]; then
    echo "Found $output; skipping download."
    return
  fi

  echo "Downloading $output..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --output "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="$output" "$url"
  else
    echo "curl or wget is required to download the dataset." >&2
    exit 1
  fi
}

extract_split() {
  local archive="$1"
  local archive_split="$2"
  local target_split="$3"
  local target_dir="$DATA_DIR/$target_split"
  local tmp_dir="$DATA_DIR/.extract-$archive_split"

  if [[ -d "$target_dir/audio" && -f "$target_dir/metadata.csv" ]]; then
    echo "Found $target_dir; skipping extraction."
    return
  fi

  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  echo "Extracting $archive to $target_dir..."
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$archive" -d "$tmp_dir"
  else
    conda run -n "$CONDA_ENV_NAME" python -m zipfile -e "$archive" "$tmp_dir"
  fi

  rm -rf "$target_dir"
  if [[ -d "$tmp_dir/$target_split" ]]; then
    mv "$tmp_dir/$target_split" "$target_dir"
  elif [[ -d "$tmp_dir/$archive_split" ]]; then
    mv "$tmp_dir/$archive_split" "$target_dir"
  elif [[ -d "$tmp_dir/audio" && -f "$tmp_dir/metadata.csv" ]]; then
    mkdir -p "$target_dir"
    shopt -s dotglob
    mv "$tmp_dir"/* "$target_dir"/
    shopt -u dotglob
  else
    echo "Could not identify the extracted layout for $archive." >&2
    echo "Expected $archive_split/, $target_split/, or files directly inside the archive." >&2
    exit 1
  fi

  rm -rf "$tmp_dir"
}

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Found Conda environment $CONDA_ENV_NAME; skipping creation."
else
  echo "Creating Conda environment $CONDA_ENV_NAME with Python $PYTHON_VERSION..."
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
fi

conda run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip
conda install -y -n "$CONDA_ENV_NAME" -c conda-forge ffmpeg
conda run -n "$CONDA_ENV_NAME" python -m pip install -r requirements.txt

mkdir -p "$DOWNLOAD_DIR"

download_file "https://zenodo.org/records/20056072/files/train.zip?download=1" "$DOWNLOAD_DIR/train.zip"
download_file "https://zenodo.org/records/20056072/files/validation.zip?download=1" "$DOWNLOAD_DIR/validation.zip"
download_file "https://zenodo.org/records/20056072/files/test.zip?download=1" "$DOWNLOAD_DIR/test.zip"

extract_split "$DOWNLOAD_DIR/train.zip" "train" "train"
extract_split "$DOWNLOAD_DIR/validation.zip" "validation" "validation"
extract_split "$DOWNLOAD_DIR/test.zip" "test" "test"

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  conda activate $CONDA_ENV_NAME"
echo
echo "Run the baseline with:"
echo "  python train.py --dataset_path $DATA_DIR --output_dir runs --experiment_name atst_f_baseline"
echo
echo "Compute annotator quality scores for Weighted Soft aggregation with:"
echo "  python compute_annotator_score.py --root $DATA_DIR"
