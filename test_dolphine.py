import subprocess
import sys
from pathlib import Path
import time

# 경로만 맞춰서 수정
DOLPHIN_REPO = Path("/home/shaush/work/Dolphin")
MODEL_PATH = DOLPHIN_REPO / "hf_model"
PDF_PATH = Path("/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf")
OUT_DIR = Path("./out")

OUT_DIR.mkdir(exist_ok=True)

t0 = time.perf_counter()

cmd = [
    sys.executable,
    "demo_page.py",
    "--model_path", str(MODEL_PATH),
    "--input_path", str(PDF_PATH),
    "--save_dir", str(OUT_DIR),
]

subprocess.run(cmd, cwd=DOLPHIN_REPO, check=True)

t1 = time.perf_counter()
print(f"Total parsing time: {t1 - t0:.2f} sec")
print("Done. Check ./out directory.")
