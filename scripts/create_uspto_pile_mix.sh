#!/bin/bash
#  This scripts creates a mix of USPTO and Pile data.
DOMAIN_FILE_PATH="../data/pile_uspto/*.jsonl.zst"
PILE_FILE_PATH="../data/pile_01/01.jsonl.zst"
OUTPUT_DIR="../data/mix_uspto_all"


# Check if python3 command is available and use it if possible
if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

# Run the command using the selected python command
$PYTHON_CMD -c "from mdel.pile_utils import create_pile_domain_mix; create_pile_domain_mix('$DOMAIN_FILE_PATH', '$PILE_FILE_PATH', '$OUTPUT_DIR')"


python -c "from mdel.pile_utils import create_pile_domain_mix; create_pile_domain_mix('$DOMAIN_FILE_PATH', '$PILE_FILE_PATH', '$OUTPUT_DIR')"
