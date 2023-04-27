#!/bin/bash
ZST_FILES="../data/mix_uspto_all/*.jsonl.zst"

if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

for file in $ZST_FILES
do
    zstd -d $file
    filename="${file%.zst}"
    $PYTHON_CMD src/mdel/pile_upload.py --file_path "$filename" --hf_repo $1
    rm $filename
done
