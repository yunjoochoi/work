#!/bin/bash
#SBATCH --job-name=docling_parser 
#SBATCH -p dell_cpu
#SBATCH --qos=cpu_qos 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16      
#SBATCH --output=logs_cpu/out_%j.out 
#SBATCH --time=02:30:00           

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"


# # thread threshold 너무올리면 속도 느려짐
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export TORCH_NUM_THREADS=1


# cd /home/shaush/work
# source .venv/bin/activate
echo "dots_ocr/parse"

cd /home/shaush/work
source .venv/bin/activate
python3 dots_ocr/parser.py \
  /home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf \
  --num_thread 16

# SBATCH -p dell_cpu
# SBATCH --qos=cpu_qos 

