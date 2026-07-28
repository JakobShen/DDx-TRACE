#!/usr/bin/env bash
set -euo pipefail

# Prepare the model-independent DDx-TRACE code and dataset.
usage() {
  echo "Usage: $0 --work-dir PATH [--code-url URL] [--data-url URL]"
}

work_dir=""
code_url="https://github.com/DDx-Trace/DDx-TRACE.git"
data_url="https://huggingface.co/datasets/Anonym001/DDx-TRACE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) work_dir="$2"; shift 2 ;;
    --code-url) code_url="$2"; shift 2 ;;
    --data-url) data_url="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$work_dir" ]] || { usage >&2; exit 2; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
git lfs version >/dev/null || { echo "git-lfs is required" >&2; exit 1; }

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
code_dir="$work_dir/DDx-TRACE"
data_dir="$work_dir/DDx-TRACE-data"

if [[ -e "$code_dir" && ! -d "$code_dir/.git" ]]; then
  echo "Refusing to overwrite non-git path: $code_dir" >&2
  exit 1
fi
if [[ ! -d "$code_dir/.git" ]]; then
  git clone "$code_url" "$code_dir"
fi

if [[ -e "$data_dir" && ! -d "$data_dir/.git" ]]; then
  echo "Refusing to overwrite non-git path: $data_dir" >&2
  exit 1
fi
if [[ ! -d "$data_dir/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$data_url" "$data_dir"
fi
git -C "$data_dir" lfs pull

archive="$data_dir/data/DDx-TRACE-images.tar.zst"
checksum="$data_dir/data/DDx-TRACE-images.tar.zst.sha256"
manifest="$data_dir/data/eurorad_neuro_01_release.json"
[[ -f "$archive" && -f "$checksum" && -f "$manifest" ]] || {
  echo "Dataset download is incomplete under $data_dir" >&2
  exit 1
}

(
  cd "$data_dir"
  sha256sum -c "data/DDx-TRACE-images.tar.zst.sha256"
)

image_count="$(find "$data_dir/images" -type f 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$image_count" == "0" ]]; then
  if command -v unzstd >/dev/null; then
    tar --use-compress-program=unzstd -xf "$archive" -C "$data_dir"
  elif command -v zstd >/dev/null; then
    tar -I zstd -xf "$archive" -C "$data_dir"
  else
    echo "Install zstd/unzstd to extract $archive" >&2
    exit 1
  fi
fi

image_count="$(find "$data_dir/images" -type f | wc -l | tr -d ' ')"
[[ "$image_count" -gt 0 ]] || { echo "No extracted images found" >&2; exit 1; }

echo "Code commit:    $(git -C "$code_dir" rev-parse HEAD)"
echo "Dataset commit: $(git -C "$data_dir" rev-parse HEAD)"
echo "Manifest:       $manifest"
echo "Image root:     $data_dir"
echo "Image files:    $image_count"
