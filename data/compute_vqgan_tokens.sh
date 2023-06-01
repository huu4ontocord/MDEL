#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:4
#SBATCH --partition=develbooster
#SBATCH --cpus-per-task=32

export CUDA_VISIBLE_DEVICES=0,1,2,3 # ensures GPU_IDs are available with correct indicies

# Args
START_SHARD="00000"
echo START_SHARD=$START_SHARD

END_SHARD="00012"
echo END_SHARD=$END_SHARD

PATHS="/p/fastdata/mmlaion/laion-400m/LAION-400m-webdataset/data/{$START_SHARD..$END_SHARD}.tar"
echo PATHS=$PATHS

OUTPUT_DIR="/p/fastdata/mmlaion/vqgan_f16_16384_laion_400M/"
echo OUTPUT_PATH=$OUTPUT_DIR

NUM_WORKERS=4
echo NUM_WORKERS=$NUM_WORKERS

NUM_GPUS=4
echo NUM_GPUS=$NUM_GPUS

MODEL_DIR="/p/scratch/ccstdl/mhatre1/hf_models/"
echo MODEL_DIR=$MODEL_DIR
# Args

source /p/project/ccstdl/gupta6/miniconda3/bin/activate
conda activate gptneox

srun --cpu-bind=v --accel-bind=gn python -u vqgan_tokens.py -p $PATHS \
				-o $OUTPUT_DIR \
				-nw $NUM_WORKERS \
				-ng $NUM_GPUS \
				-md $MODEL_DIR

# python -u : produce output immediately, no buffer caching
#srun --cpu-bind=v --accel-bind=gn  python -u dummy_script.py