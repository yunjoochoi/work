import time
from pathlib import Path

# ✅ 레이아웃 모드: 있으면 켜고, 없으면 조용히 스킵
LAYOUT_ENABLED = True
try:
    import pymupdf.layout  # pip install pymupdf-layout
    LAYOUT_ENABLED = True
except Exception as e:
    print(f"[WARN] pymupdf.layout not available (layout mode OFF): {e}")

import pymupdf4llm


def pdf_to_md_with_images(pdf_path: str, out_dir: str, dpi: int = 150):
    t_all = time.perf_counter()

    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{pdf_path.stem}.md"

    print(f"[INFO] layout mode: {'ON' if LAYOUT_ENABLED else 'OFF'}")
    print(f"[INFO] input:  {pdf_path}")
    print(f"[INFO] output: {md_path}")
    print(f"[INFO] images: {images_dir}")

    # ---- 변환 시간 측정 ----
    t_conv = time.perf_counter()
    md_text = pymupdf4llm.to_markdown(
        str(pdf_path),

        # 이미지/그래픽 저장 + md 링크 삽입
        write_images=True,
        image_path=str(images_dir),
        image_format="png",
        dpi=dpi,

        # 표 -> 마크다운
        table_strategy="lines_strict",

        # 레이아웃/노이즈
        page_separators=True,
        header=True,
        footer=True,

        # OCR (스캔 PDF가 아니면 보통 OFF가 빠름)
        use_ocr=False,
        ocr_dpi=400,

        show_progress=True,
    )
    conv_sec = time.perf_counter() - t_conv

    # ---- 저장 시간 측정 ----
    t_save = time.perf_counter()
    md_path.write_text(md_text, encoding="utf-8")
    save_sec = time.perf_counter() - t_save

    total_sec = time.perf_counter() - t_all

    print(f"✅ MD saved: {md_path}")
    print(f"✅ Images saved under: {images_dir}")
    print(f"[TIME] convert: {conv_sec:.2f}s | save: {save_sec:.2f}s | total: {total_sec:.2f}s")


if __name__ == "__main__":
    pdf_to_md_with_images(
        pdf_path="/home/shaush/pdfs/2025년+8월+산업활동동향+보도자료.pdf",
        out_dir="/home/shaush/work/",
        dpi=150,
    )
