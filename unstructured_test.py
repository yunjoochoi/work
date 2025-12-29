import time
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Image, Table, Formula
)

def convert_pdf_to_markdown_with_assets(
    pdf_path: str, 
    output_dir: str = "output",
    korean_support: bool = True
):
    # 1. 시간 측정 시작
    start_time = time.time()
    
    # 이미지 저장 경로 설정
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Processing '{pdf_path}'...")

    # 2. PDF 파티셔닝 (핵심 로직)
    # strategy="hi_res": 이미지/수식/표 구조 인식을 위해 필수 (Vision 모델 가동)
    # languages=["kor"]: 한국어 OCR 지원
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",                                     # 이미지/수식 인식을 위해 hi_res 필수
            languages=["kor", "eng"] if korean_support else ["eng"], 
            extract_images_in_pdf=True,                            # 이미지 추출 활성화
            extract_image_block_types=["Image", "Table"],          # 저장할 요소 타입
            extract_image_block_output_dir=images_dir,             # 이미지 저장 경로
            infer_table_structure=True,                            # 표 구조 인식 활성화
        )
    except Exception as e:
        print(f"Error during partitioning: {e}")
        return

    # 3. 마크다운 변환 및 페이지/플레이스홀더 처리
    markdown_lines = []
    
    for element in elements:
        # 5. 페이지 넘버 (메타데이터에서 추출)
        page_num = element.metadata.page_number
        page_tag = f" " if page_num else ""
        
        text = str(element)
        md_text = ""

        # 요소 타입별 마크다운 포맷팅
        if isinstance(element, Title):
            md_text = f"## {text}"
        
        elif isinstance(element, ListItem):
            md_text = f"- {text}"
            
        elif isinstance(element, Formula):
            # 4. 수식 인식 (LaTeX로 변환되거나 텍스트로 추출됨)
            # unstructured의 수식 모델이 완벽하지 않을 수 있어 $$ 감싸기 시도
            md_text = f"$$ {text} $$"
            
        elif isinstance(element, Table):
            # 표는 HTML로 변환된 것을 가져오거나 텍스트 사용
            if element.metadata.text_as_html:
                md_text = element.metadata.text_as_html
            else:
                md_text = f"```\n{text}\n```"

        elif isinstance(element, Image):
            # 3. 이미지 플레이스홀더 (저장된 경로 사용)
            image_path = element.metadata.image_path
            if image_path:
                # 절대 경로 대신 마크다운 파일 기준 상대 경로로 표시하는 것이 일반적
                rel_path = os.path.relpath(image_path, output_dir)
                md_text = f"![Extracted Image]({rel_path})"
            else:
                md_text = "![Image Detected but not saved]"
        
        else:
            # NarrativeText 및 기타
            md_text = text

        # 결과 리스트에 추가 (내용 + 페이지 태그)
        if md_text.strip():
            markdown_lines.append(md_text + page_tag)

    # 마크다운 파일 저장
    md_output_path = os.path.join(output_dir, "result.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(markdown_lines))

    # 2. 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n=== 완료 ===")
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    print(f"마크다운 저장: {md_output_path}")
    print(f"이미지 저장: {images_dir}")

# --- 실행 예시 ---
# 사용 시 실제 PDF 경로로 변경해주세요.
convert_pdf_to_markdown_with_assets("/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf")