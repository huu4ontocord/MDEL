#!/bin/bash
#  This scripts creates a mix of USPTO and Pile data.
# Check if python3 command is available and use it if possible
if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

SUBSET_NAME="USPTO Backgrounds"

for SPLIT in "test" "val" "train"
do
  PILE_FILE_PATH="../data/pile/$SPLIT/*.jsonl.zst"
  OUTPUT_DIR="../data/mix_uspto_all/$SPLIT"

  # Shard the input data
  $PYTHON_CMD -c "from mdel.pile_utils import *; split_pile('$PILE_FILE_PATH')"

  $PYTHON_CMD -c "from mdel.pile_utils import *; create_pile_domain_mix('$PILE_FILE_PATH', '$PILE_FILE_PATH', '$OUTPUT_DIR', '$SUBSET_NAME')"
done
