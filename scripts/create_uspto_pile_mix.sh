#!/bin/bash
#  This scripts creates a mix of USPTO and Pile data.
# Check if python3 command is available and use it if possible
if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

for SPLIT in "test" "val" "train"
do
  PILE_FILE_PATH="../$SPLIT/test/*.jsonl.zst"
  $PYTHON_CMD -c "from mdel.pile_utils import *; create_pile_domain_mix('$PILE_FILE_PATH', '$PILE_FILE_PATH', '$OUTPUT_DIR')"
done
