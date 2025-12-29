import time
import os
from markitdown import MarkItDown

# OpenAI 관련 라이브러리(import openai) 및 설정 제거됨

def convert_plain_no_client(input_path, output_path):
    # 1. 파일 존재 여부 확인
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return

    print(f"🔄 변환 시작 (Plain Mode): {input_path}")
    
    # 2. 시간 측정 시작
    start_time = time.time()

    try:
        # 3. MarkItDown 초기화 (인자 없이 호출)
        # - LLM 클라이언트를 연결하지 않았으므로 순수 텍스트 추출 엔진만 작동합니다.
        md = MarkItDown()

        # 4. 변환 실행
        result = md.convert(input_path)
        
        # 5. 결과 저장 (UTF-8)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.text_content)
            
        print(f"✅ 변환 완료 및 저장: {output_path}")

    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")

    # 6. 시간 측정 종료 및 출력
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️ 총 소요 시간: {elapsed:.4f}초")

if __name__ == "__main__":
    input_file = "/home/shaush/pdf/2025년+8월+산업활동동향+보도자료.pdf"     # 변환할 파일명
    output_file = "markitdown.md" # 저장할 파일명
    
    convert_plain_no_client(input_file, output_file)