# doc = fitz.open("20251128_company_828658000.pdf")
# page = doc[0] 

# raw_content = page.read_contents()

# decoded_content = raw_content.decode("utf-8", errors="ignore")

# print(decoded_content[:500])
import fitz
import pymupdf4llm

pdf_path = "/home/shaush/pdfs/default.pdf"
doc = fitz.open(pdf_path)

out = []
for i in range(doc.page_count):
    out.append(pymupdf4llm.to_markdown(doc, pages=[i]))

output_path= "fitz_test.md"
with open(output_path, "w", encoding="utf-8") as f:
    for i, page_md in enumerate(out):
        f.write(f"\n\n---\n## Page {i+1}\n\n")
        f.write(page_md)
