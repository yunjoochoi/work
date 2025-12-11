# import fitz  # pip install pymupdf

# def is_mostly_scanned_pdf_fitz(
#     data: bytes,
#     max_sample_pages: int = 10,
#     min_text_pages_for_text_pdf: int = 1,
# ) -> bool:
#     """
#     Sample pages in a PDF and return True if it looks like a scanned (image-based) document
#     and False if it looks like a text-based document.

#     Parameters
#     ----------
#     data : bytes
#         Raw PDF bytes (e.g. from `open(path, "rb").read()`).
#     max_sample_pages : int, optional
#         Maximum number of pages to sample across the document (upper bound).
#     min_text_pages_for_text_pdf : int, optional
#         Minimum number of sampled pages that must contain text to treat the PDF as text-based.
#     """
#     doc = fitz.open(stream=data, filetype="pdf")
#     num_pages = doc.page_count
#     if num_pages == 0:
#         return True

#     # Sample up to `max_sample_pages` pages, evenly spaced
#     step = max(1, num_pages // max_sample_pages)
#     text_pages = 0
#     sampled = 0

#     for i in range(0, num_pages, step):
#         page = doc.load_page(i)
#         text = page.get_text("text") or ""
#         if text.strip():
#             text_pages += 1
#             if text_pages >= min_text_pages_for_text_pdf:
#                 return False

#         sampled += 1
#         if sampled >= max_sample_pages:
#             break

#     return True
 

# import pymupdf
# import time
# import io

# file_name="/home/coder/project/yjchoi/docling_parser/temp/과제4_11주차_2025311605_최윤주.docx"
# with open(file_name, "rb") as f:
#     pdf_bytes=f.read()


# out=is_mostly_scanned_pdf_fitz(io.BytesIO(pdf_bytes))
# print(out)

# ---------------------------------------------------------------------


import fitz  # pip install pymupdf
import pymupdf4llm  # pip install pymupdf4llm
import io
import fitz  # pip install pymupdf
import pymupdf4llm  # pip install pymupdf4llm

def is_mostly_scanned_pdf_pymupdf4llm(
    data: bytes,
    max_sample_pages: int = 10,
    min_text_pages_for_text_pdf: int = 1,
) -> bool:
    """
    Apply fitz sampling logic (evenly spaced) to pymupdf4llm extraction.
    """
    doc = fitz.open(stream=data, filetype="pdf")
    num_pages = doc.page_count
    
    if num_pages == 0:
        return True

    step = max(1, num_pages // max_sample_pages)
    
    found_text_pages = 0
    sampled = 0

    for i in range(0, num_pages, step):
        md_text = pymupdf4llm.to_markdown(doc, pages=[i])
        
        if md_text.strip():
            found_text_pages += 1
            if found_text_pages >= min_text_pages_for_text_pdf:
                return False

        sampled += 1
        if sampled >= max_sample_pages:
            break

    return True

# file_name="/home/coder/project/yjchoi/docling_parser/not3.pdf"
# with open(file_name, "rb") as f:
#     pdf_bytes=f.read()

from pathlib import Path
import time

folder=Path("/home/shaush/pdfs")

li=[p.resolve() for p in folder.iterdir()]

for file in li:
    with open(file, "rb") as f:
        pdf_bytes=f.read()
    start_time = time.perf_counter()
    out=is_mostly_scanned_pdf_pymupdf4llm(io.BytesIO(pdf_bytes))
    print(time.perf_counter() - start_time)
    print(file, out)


