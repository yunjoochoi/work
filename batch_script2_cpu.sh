#!/bin/bash
#SBATCH --job-name=docling_parser 
#SBATCH -p dell_cpu
#SBATCH --qos=cpu_qos 
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32      
#SBATCH --output=logs_cpu/out_%j.out 
#SBATCH --time=02:30:00           

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "/home/shaush/work/no_cpu_set_mm_doc_tool_docbatch_fallback_figureclass_basemodel_quantization.py"


# # thread threshold 너무올리면 속도 느려짐
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export TORCH_NUM_THREADS=1


cd /home/shaush/work
source .venv/bin/activate
python /home/shaush/work/no_cpu_set_mm_doc_tool_docbatch_fallback_figureclass_basemodel_quantization.py

# SBATCH -p dell_cpu
# SBATCH --qos=cpu_qos 

