#!/bin/bash
#SBATCH --job-name=docling_parser 
#SBATCH -p dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16        
#SBATCH --output=logs_gpu/out_%j.out 
#SBATCH --time=01:30:00           

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
# echo "/home/shaush/work/mm_doc_tool_docbatch_fallback_figureclass_basemodel.py"
# echo "GPU script w/o limit"


echo "Dolphin"

# cd /home/shaush/work
# source .venv/bin/activate

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate deepseek-ocr
cd /home/shaush/work/Dolphin
source .venv/bin/activate
# python3 dots_ocr/parser.py demo/demo_pdf1.pdf  --num_thread 64

# python /home/shaush/DeepSeek-OCR/test_deepseek_ocr.py   --pdf "/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf"   --out_dir "./dsocr_vllm_out"   --scale 2.0   --batch_size 4   --max_tokens 2048
python /home/shaush/work/Dolphin/demo_page_.py
# python /home/shaush/work/Dolphin/demo_page.py --input_path /home/shaush/pdf/2025년+8월+산업활동동 향+보도자료.pdf
# python /home/shaush/MonkeyOCR/parse.py
# python /home/shaush/work/unstructured_test.py
# python /home/shaush/work/mm_doc_tool_docbatch_fallback_figureclass_basemodel.py
# python granite-docling.py \
#   --pdf /home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf \
#   --out_dir ./out \

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
