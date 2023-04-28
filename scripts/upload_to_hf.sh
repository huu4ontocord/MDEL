#!/bin/bash
if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi


ZST_FILES="../data/mix_uspto_all/val/*.jsonl.zst"
$PYTHON_CMD src/mdel/pile_upload.py --file_path "$ZST_FILES" --hf_repo $1 --split val

ZST_FILES="../data/mix_uspto_all/test/*.jsonl.zst"
$PYTHON_CMD src/mdel/pile_upload.py --file_path "$ZST_FILES" --hf_repo $1 --split test

ZST_FILES="../data/mix_uspto_all/train/*.jsonl.zst"
$PYTHON_CMD src/mdel/pile_upload.py --file_path "$ZST_FILES" --hf_repo $1 --split train
