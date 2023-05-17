#!/bin/bash

if command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

for MODEL in "Multi-Domain-Expert-Layers/expert-arxiv" "Multi-Domain-Expert-Layers/expert-freelaw" "Multi-Domain-Expert-Layers/expert-github" "EleutherAI/pythia-1b-deduped"
do
  for DATASET in "Multi-Domain-Expert-Layers/arxiv" "Multi-Domain-Expert-Layers/freelaw" "Multi-Domain-Expert-Layers/github"
  do
    for SPLIT in "validation_domain" "train" "validation_pile"
    do
      JOB_NAME="${MODEL}-${DATASET}-${SPLIT}"
      sbatch --job-name="$JOB_NAME" <<EOT
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=small-g
#SBATCH --account=project_462000259
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=small-g

echo RUNNING $MODEL on $DATASET with $SPLIT
rocm-smi
export WANDB_MODE=offline
export HF_HOME="/scratch/project_462000259/jstillerman/hf_cache"
$PYTHON_CMD ../../src/mdel/calculate_perplexity.py --model $MODEL --dataset $DATASET --split $SPLIT
EOT
    done
  done
done
