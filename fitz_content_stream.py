import fitz
import pymupdf4llm
# doc = fitz.open("20251128_company_828658000.pdf")
# page = doc[0]  # 1페이지

# # 1. 원시 바이트 형태의 Content Stream 읽기
# raw_content = page.read_contents()

# # 2. 사람이 읽을 수 있는 문자열로 디코딩
# decoded_content = raw_content.decode("utf-8", errors="ignore")

# # 3. 앞부분 500자만 출력해보기
# print(decoded_content[:500])
doc = fitz.open("/data/yunju/pdfs/20251128_company_828658000.pdf")
text = ""

for page in doc:
    # "markdown" 옵션 사용
    text += page.get_text("markdown")

print(text)