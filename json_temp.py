import fitz

doc = fitz.open("/data/yunju/pdfs/not3.pdf")
page = doc[0]  # 주가 추이 그래프가 있는 페이지

# 이미지 목록 조회
image_list = page.get_images()

print(f"발견된 이미지 개수: {len(image_list)}")
print(image_list)