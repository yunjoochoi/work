#!/bin/bash
#SBATCH --job-name=docling_parser 
#SBATCH -p suma_A6000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16        
#SBATCH --output=logs/out_%j.out 
#SBATCH --error=logs/err_%j.err
#SBATCH --time=00:30:00           


echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "64"


# thread threshold (CPU)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

cd work
source .venv/bin/activate
python -u mm_doc_tool_docbatchver2.py

# SBATCH -p dell_cpu
# SBATCH --qos=cpu_qos 

# SBATCH -p suma_a100
# SBATCH -q a100_qos

# SBATCH -p suma_A6000
# SBATCH --gres=gpu:4