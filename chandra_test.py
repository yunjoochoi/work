# test_chandra.py
import subprocess
import time
from pathlib import Path

PDF = Path("/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf")
OUT_DIR = Path("./output")

t0 = time.perf_counter()
subprocess.run(
    ["chandra", str(PDF), str(OUT_DIR), "--method", "hf"],
    check=True,
)
t1 = time.perf_counter()

print(f"Done. elapsed_sec={t1 - t0:.3f}")
print(f"Output dir: {OUT_DIR}")
