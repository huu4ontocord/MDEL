#!/usr/bin/env bash
# SLURM Configuration
#SBATCH --account=cstdl
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition develbooster


# !!! change to the log directory !!!

#SBATCH --output=/p/project/ccstdl/persaud1/logs/%j_0_log.out

# JUWELS Configuration
ml --force purge
ml use $OTHERSTAGES
ml Stages/2023
ml GCC
ml OpenMPI
ml CUDA
ml cuDNN
ml NCCL
ml git
ml PyTorch
ml torchvision
module unload PyTorch 
module unload torchvision


# Network Configuration
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_ASYNC_ERROR_HANDLING=1


echo $SLURM_JOB_GPUS
echo $SLURM_NTASKS
echo $SLURM_NODELIST

# Convert SLURM_JOB_GPUS to an array
IFS=',' read -ra GPU_ARRAY <<< "$SLURM_JOB_GPUS"

# Get the number of GPUs from the length of the array
NUM_GPUS=${#GPU_ARRAY[@]}

export TOTAL_GPUS=$(($NUM_GPUS * $SLURM_NTASKS))
echo $TOTAL_GPUS

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr

# Environment Configuration
source $SCRATCH_ccstdl/persaud1/mdelVenv/bin/activate


# !!! change to the cache directory !!!

cache_dir="/p/scratch/ccstdl/persaud1/hf_cache"


export TRANSFORMERS_CACHE=$cache_dir
export HF_DATASETS_CACHE=$cache_dir
export HF_HOME=$cache_dir



export PATH="/p/software/juwelsbooster/stages/2023/software/OpenMPI/4.1.4-GCC-11.3.0/bin:$PATH"

# Juwels specific env

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_MODE="offline"
export TRANSFORMERS_OFFLINE=1


export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`


# Print System Information
echo "GPUs available to job: $SLURM_JOB_GPUS"
echo "Total tasks: $SLURM_NTASKS"



# Loop over all nodes
for ((i=0; i<$COUNT_NODE; i++))
do
    srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n "$((i+1))p")" \
    torchrun --master_addr "$MASTER_ADDR" --master_port 12802 --node_rank $i \
             --nnodes $SLURM_NTASKS \
             --nproc-per-node=$NUM_GPUS \
             /p/home/jusers/persaud1/juwels/persaud1/scaled-rope/finetune.py \
             --output_dir saved_ckpts_32k \
             --configs defaults lora-70b-llama2 \
             --deepspeed &
done

wait