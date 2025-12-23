#!/bin/bash
#SBATCH --job-name=docling_parser 
#SBATCH -p dell_rtx3090
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8        
#SBATCH --output=logs_gpu/out_%j.out 
#SBATCH --time=00:30:00           


echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "/home/shaush/work/no_cpu_set_mm_doc_tool_docbatch_fallback_figureclass_basemodel_quantization.py"
echo "GPU script w/o limit"

cd /home/shaush/work
source .venv/bin/activate
python /home/shaush/work/no_cpu_set_mm_doc_tool_docbatch_fallback_figureclass_basemodel_quantization.py

# SBATCH -p dell_cpu
# SBATCH --qos=cpu_qos 

# SBATCH -p suma_a100
# SBATCH -q a100_qos

# SBATCH -p suma_A6000
# SBATCH --gres=gpu:4

# SBATCH -p dell_rtx3090
# SBATCH -p suma_rtx4090


# # thread threshold (CPU)
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# export TORCH_NUM_THREADS=4
