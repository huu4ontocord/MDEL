#!/bin/bash
if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

HF_REPO=Multi-Domain-Expert-Layers/uspto

for SPLIT in "test" "val" "train"
do
  FOLDER_PATH=$(readlink -f ../data/mix_uspto_all/$SPLIT/)
  $PYTHON_CMD src/mdel/pile_upload.py --folder-path "$FOLDER_PATH" --hf-repo $HF_REPO --split $SPLIT
done
