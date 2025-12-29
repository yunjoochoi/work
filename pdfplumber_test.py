import pdfplumber
import pandas as pd
import time
import os

# ==========================================
# 1. 설정 (경로 지정)
# ==========================================
pdf_path = "/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf"
output_dir = "/home/shaush/work/parsed_outputs"  # 결과 저장할 폴더
os.makedirs(f"{output_dir}/images", exist_ok=True) # 이미지 저장 폴더 생성

# ==========================================
# 2. 실행 및 시간 측정 시작
# ==========================================
print(f"Processing: {pdf_path}")
start_time = time.time()

full_text_content = ""
full_table_content = ""
image_count = 0

try:
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total Pages: {total_pages}")

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            print(f"Processing Page {page_num}/{total_pages}...", end="\r")
            
            # --------------------------------------
            # A. 텍스트 추출 (Text)
            # --------------------------------------
            text = page.extract_text()
            if text:
                full_text_content += f"\n\n=== Page {page_num} ===\n{text}"

            # --------------------------------------
            # B. 표 추출 (Tables -> Markdown)
            # --------------------------------------
            # extract_tables()는 페이지 내 모든 표를 리스트로 반환
            tables = page.extract_tables() 
            
            if tables:
                full_table_content += f"\n\n## Page {page_num} Tables\n"
                for idx, table in enumerate(tables):
                    # 데이터가 있고 헤더가 있는 경우만 처리
                    if table and len(table) > 1:
                        try:
                            # None 값은 빈 문자열로 대체하여 오류 방지
                            cleaned_table = [['' if item is None else item for item in row] for row in table]
                            
                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                            markdown_table = df.to_markdown(index=False)
                            
                            full_table_content += f"\n### Table {idx+1}\n{markdown_table}\n"
                        except Exception as e:
                            print(f"\n[Error] Table processing failed on p{page_num}: {e}")

            # --------------------------------------
            # C. 이미지 추출 (Images -> Save PNG)
            # --------------------------------------
            # pdfplumber는 이미지 객체 정보를 리스트로 반환 (좌표 포함)
            for img_idx, img in enumerate(page.images):
                # 너무 작은 이미지(아이콘, 선 등)는 건너뛰기 (가로/세로 50px 이하 필터링)
                if img['width'] < 50 or img['height'] < 50:
                    continue

                try:
                    # 이미지의 좌표(Bounding Box) 가져오기
                    bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                    
                    # 해당 영역 잘라내기 (Crop)
                    cropped_page = page.crop(bbox)
                    
                    # 이미지 파일로 변환 (해상도 200dpi) 및 저장
                    im_obj = cropped_page.to_image(resolution=200)
                    img_filename = f"{output_dir}/images/p{page_num}_img{img_idx+1}.png"
                    im_obj.save(img_filename)
                    image_count += 1
                except Exception as e:
                    # 크롭 영역이 페이지를 벗어나는 등의 오류 무시
                    pass

    # ==========================================
    # 3. 결과 파일 저장
    # ==========================================
    # 텍스트 저장
    with open(f"{output_dir}/extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text_content)
    
    # 마크다운 표 저장
    with open(f"{output_dir}/extracted_tables.md", "w", encoding="utf-8") as f:
        f.write(full_table_content)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n\n[Done] Processing Completed!")
    print(f"- Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"- Text saved to: {output_dir}/extracted_text.txt")
    print(f"- Tables saved to: {output_dir}/extracted_tables.md")
    print(f"- Images extracted: {image_count} (saved in {output_dir}/images/)")

except Exception as e:
    print(f"\n[Critical Error] Failed to process PDF: {e}")