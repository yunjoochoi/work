from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False 
pipeline_options.do_table_structure = True


from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend 
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            # backend=PyPdfiumDocumentBackend
        )
    }
)

file_path = "/home/shaush/work/sample/default.pdf"
print(f"Processing with PyPdfium2 Backend: {file_path}")

with open("PyPdfiumDocumentBackend_output.md", "w", encoding="utf-8") as f:
    f.write(md_text)