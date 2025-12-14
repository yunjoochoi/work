import pikepdf

src = pikepdf.open("/home/shaush/pdfs/2025년+8월+산업활동동향+보도자료.pdf")

dst = pikepdf.Pdf.new()

# 2번째 페이지부터 끝까지
for page in src.pages[1:]:
    dst.pages.append(page)

# 첫 페이지를 마지막에
dst.pages.append(src.pages[0])

dst.save("rearranged.pdf")
