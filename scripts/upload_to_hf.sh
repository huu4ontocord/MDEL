#!/bin/bash
if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

HF_REPO=Multi-Domain-Expert-Layers/uspto
for SPLIT in "test" "val" "train"
do
  ZST_FILES="../data/mix_uspto_all/'$SPLIT'/*.jsonl.zst"
  $PYTHON_CMD src/mdel/pile_upload.py --file_path "$ZST_FILES" --hf_repo $HF_REPO --split $SPLIT
done
