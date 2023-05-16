#!/bin/bash

if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

SUBSET_NAME="USPTO Backgrounds"

for MODEL in "Multi-Domain-Expert-Layers/expert-arxiv" "Multi-Domain-Expert-Layers/expert-freelaw" "Multi-Domain-Expert-Layers/expert-github" "EleutherAI/pythia-1b-deduped"
do
  for DATASET in "Multi-Domain-Expert-Layers/arxiv" "Multi-Domain-Expert-Layers/freelaw" "Multi-Domain-Expert-Layers/github"
  do
    for SPLIT in "validation_domain" "train" "validation_pile"
    do
      $PYTHON_CMD ../src/mdel/calculate_perplexity.py --model $MODEL --dataset $DATASET --split $SPLIT
    done
  done
done
