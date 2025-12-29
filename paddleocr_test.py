import os
import time
import fitz  # PyMuPDF
import cv2
import numpy as np
from paddleocr import PPStructureV3

def pdf_to_markdown_final(pdf_path, output_root="output"):
    # 0. 시간 측정 시작
    start_time = time.time()
    
    # 결과 저장 루트 디렉토리 생성
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print("[Init] Loading PaddleOCR PP-StructureV3 Model...")
    
    # 1. 모델 초기화 (에러를 유발하던 show_log 제거)
    # 공식 코드 스니펫의 설정을 따름
    # lang='korean'은 한국어 인식을 위해 필수이므로 유지 (만약 이것도 에러나면 제거해야 함)
    try:
        pipeline = PPStructureV3(
            lang='korean',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
    except ValueError as e:
        print(f"[Warning] 'lang' argument might be invalid in this version. Retrying without it. ({e})")
        # lang 인자까지 에러가 날 경우를 대비한 Fallback
        pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )

    # 2. PDF 파일 열기
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[Info] Processing {pdf_path} ({total_pages} pages)...")

    for page_num, page in enumerate(doc):
        current_page = page_num + 1
        print(f"  - Processing Page {current_page}/{total_pages}...")
        
        # 3. PDF -> 이미지 변환 (해상도 2배)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 4. 공식 Predict 호출
        # input에 이미지 배열(numpy)을 직접 전달
        output = pipeline.predict(input=img)

        # 5. 결과 저장 (공식 메서드 사용)
        # 페이지별로 결과가 덮어씌워지지 않도록 하위 폴더 분리
        page_save_path = os.path.join(output_root, f"page_{current_page}")
        if not os.path.exists(page_save_path):
            os.makedirs(page_save_path)
        
        for res in output:
            # 공식 기능: 마크다운으로 저장 (이미지, 엑셀 등 자동 포함)
            # save_path에 '폴더 경로'를 주면 그 안에 md 파일 생성됨
            res.save_to_markdown(save_path=page_save_path)

    # 6. 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 30)
    print(f"[Done] All pages processed.")
    print(f"       Results saved in: {output_root}/page_N/")
    print(f"[Time] Total Parsing Time: {elapsed_time:.2f} seconds")
    print("-" * 30)

if __name__ == "__main__":
    # 경로 수정 없이 바로 실행 가능하도록 설정
    pdf_path = "/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf"
    
    if os.path.exists(pdf_path):
        pdf_to_markdown_final(pdf_path)
    else:
        print(f"Error: File not found at {pdf_path}")