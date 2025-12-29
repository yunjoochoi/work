import os
import time
import re
from typing import List, Tuple

import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================
# Configuration
# =========================
MODEL_PATH = "infly/Infinity-Parser-7B"
CACHE_DIR = "/scratch2/shaush/models"

# Performance Knobs
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1536 * 28 * 28  # 속도/메모리 균형값
BATCH_PAGES = 1              # VRAM 여유 시 증가 (A100이면 2~4 추천)
MAX_NEW_TOKENS = 2048
USE_FLASH_ATTN2 = True

# Prompt Engineering (수식, 이미지 처리 요청)
# Infinity-Parser는 수식 변환(LaTeX)에 특화되어 있습니다.
PROMPT = (
    "Transcribe the content of this page into Markdown format.\n"
    "1. Write mathematical formulas in LaTeX format (enclose inline math in $...$ and display math in $$...$$).\n"
    "2. If there are figures or images, keep their context but do not describe them in too much detail. "
    "Use a placeholder like `![Figure](extracted_images/page_X_img_Y.png)` if applicable.\n"
    "3. Preserve headings and tables structure."
)

def extract_and_save_images(pdf_path: str, output_dir: str, min_size: int = 2000) -> dict:
    """
    PDF에서 직접 이미지를 추출하여 저장합니다.
    Args:
        min_size: 너무 작은 아이콘/선 등을 무시하기 위한 바이트 크기 임계값
    Returns:
        페이지별 이미지 경로 리스트 딕셔너리 {page_idx: [path1, path2, ...]}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    doc = fitz.open(pdf_path)
    page_image_map = {}
    
    print(f"Extracting images from PDF to '{output_dir}'...")
    
    for i in range(len(doc)):
        page = doc.load_page(i)
        image_list = page.get_images(full=True)
        saved_images = []
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 너무 작은 이미지는 건너뜀 (노이즈 제거)
            if len(image_bytes) < min_size:
                continue
                
            image_ext = base_image["ext"]
            image_filename = f"page_{i+1:03d}_img_{img_idx+1:02d}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # 마크다운에 삽입하기 위해 상대 경로 저장 (필요시 절대경로 수정 가능)
            saved_images.append(image_filename)
            
        if saved_images:
            page_image_map[i] = saved_images
            
    doc.close()
    return page_image_map

def load_pdf_as_pil_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """PDF 페이지를 VLM 입력용 PIL 이미지로 변환"""
    doc = fitz.open(pdf_path)
    images = []
    # dpi 200은 텍스트 인식에 충분한 가성비 해상도
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def build_messages_batch(images: List[Image.Image], prompt: str) -> List[List[dict]]:
    """Qwen-VL용 메시지 구조 생성"""
    batch_messages = []
    for img in images:
        batch_messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ])
    return batch_messages

@torch.inference_mode()
def parse_pdf(
    pdf_path: str,
    output_md_path: str,
    image_save_dir: str = None
):
    # ==========================================
    # 1. 초기화 및 모델 로드
    # ==========================================
    total_start_time = time.perf_counter()
    
    print(f"Processing: {pdf_path}")
    
    # 1-1. 이미지 추출 (요구사항 3)
    extracted_images_map = {}
    if image_save_dir:
        extract_start = time.perf_counter()
        extracted_images_map = extract_and_save_images(pdf_path, image_save_dir)
        print(f"Image extraction time: {time.perf_counter() - extract_start:.2f}s")

    # 1-2. 모델 로드
    print("Loading VLM model...")
    model_load_start = time.perf_counter()
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        # attn_implementation="flash_attention_2" if USE_FLASH_ATTN2 else "eager",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    print(f"Model loaded in {time.perf_counter() - model_load_start:.2f}s")

    # ==========================================
    # 2. PDF 페이지 로드 (VLM 입력용)
    # ==========================================
    load_start = time.perf_counter()
    pil_images = load_pdf_as_pil_images(pdf_path, dpi=200)
    print(f"Loaded {len(pil_images)} pages. ({time.perf_counter() - load_start:.2f}s)")

    # ==========================================
    # 3. 추론 (Batch Processing)
    # ==========================================
    inference_start = time.perf_counter()
    page_markdowns = []
    
    # Batch 처리 헬퍼
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    total_batches = (len(pil_images) + BATCH_PAGES - 1) // BATCH_PAGES
    
    for b_idx, batch_imgs in enumerate(chunker(pil_images, BATCH_PAGES), start=1):
        # 메시지 구성
        batch_msgs = build_messages_batch(batch_imgs, PROMPT)
        
        # Preprocessing
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_msgs
        ]
        
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # Deterministic (재현성 및 속도)
        )

        # Post-processing output
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            trimmed_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        page_markdowns.extend(output_texts)
        print(f"Batch [{b_idx}/{total_batches}] completed.")

    print(f"Inference time: {time.perf_counter() - inference_start:.2f}s")

    # ==========================================
    # 4. 결과 병합 및 저장 (요구사항 1, 4, 5)
    # ==========================================
    final_md_lines = []
    
    for i, md_text in enumerate(page_markdowns):
        page_num = i + 1
        
        # 페이지 구분 헤더 (요구사항 5)
        final_md_lines.append(f"\n\n")
        final_md_lines.append(f"## Page {page_num}\n")
        
        # 텍스트 추가
        final_md_lines.append(md_text)
        
        # 해당 페이지에서 추출된 이미지가 있다면 하단에 부록으로 추가 (요구사항 3 보완)
        if i in extracted_images_map:
            final_md_lines.append("\n\n**Extracted Figures:**\n")
            for img_file in extracted_images_map[i]:
                # 이미지 경로를 상대경로로 기입
                rel_path = os.path.join(os.path.basename(image_save_dir), img_file)
                final_md_lines.append(f"![Figure]({rel_path})")
        
        final_md_lines.append("\n\n---\n")

    full_markdown = "\n".join(final_md_lines)

    # 파일 저장
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    total_time = time.perf_counter() - total_start_time
    print("="*40)
    print(f"Done! Markdown saved to: {output_md_path}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    # 경로 설정
    INPUT_PDF = "/home/shaush/pdfs/2025년+8월+산업활동동향+보도자료.pdf"
    OUTPUT_MD = "/home/shaush/work/inf_parsed_output.md"
    # 이미지가 저장될 폴더 (마크다운 파일과 같은 위치 혹은 하위 폴더 권장)
    IMAGE_DIR = "/home/shaush/work/extracted_images"

    parse_pdf(INPUT_PDF, OUTPUT_MD, IMAGE_DIR)