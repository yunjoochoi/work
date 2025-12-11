import os
import sys
from typing import Literal, Dict, Any, Optional, List, Tuple
from pathlib import Path
from io import BytesIO
import shutil
import tempfile
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, ThreadedPdfPipelineOptions
from docling_core.types.io import DocumentStream
from docling_core.types.doc.base import ImageRefMode

class DoclingParser:
    """
    Parse documents to Markdown using Docling.
    Supports PDF, DOCX, PPTX, XLSX with page-wise processing and image extraction.
    """
    def __init__(self, output_base_dir: Optional[str] = None):
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.generate_picture_images = True
        # opts.generate_table_images = False -> deprecated
        opts.images_scale = 2.0
        opts.layout_batch_size = 8 # detect regions - table or text ?
        opts.table_batch_size = 8 # 표 구조 인식 모델 TableFormer
        # opts = ThreadedPdfPipelineOptions(generate_table_images = False, generate_picture_images = True, do_ocr = False)
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
            }
        )
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path("docling_parsed_output")
        self.output_base_dir.mkdir(exist_ok=True)

    def parse(
        self,
        display_name: str,
        file_obj: BytesIO,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Parse document to Markdown.

        Args:
            display_name: Full filename with extension 
            file_obj: File object 
            output_dir: Directory to save final markdown and images (default: {output_base_dir}/{filename})
            cache_dir: Directory for temporary files (default: same as output_dir)

        Returns:
            Dict with 'format', 'markdown', 'output_dir', 'num_pages'
        """
        # TODO:
        start_time = time.time()

        path_obj = Path(display_name)
        filename = path_obj.stem

        # Setup output directory: {base_dir}/{filename}/
        base_dir = Path(output_dir) if output_dir else self.output_base_dir
        file_output_dir = base_dir / filename
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # Setup cache directory
        file_cache_dir = Path(cache_dir) if cache_dir else file_output_dir
        file_cache_dir.mkdir(parents=True, exist_ok=True)

        # Convert document
        doc_stream = DocumentStream(name=display_name, stream=file_obj)
        result = self.converter.convert(doc_stream)
        doc = result.document
        num_pages = doc.num_pages()

        # Process content
        temp_files_to_delete: List[Path] = []
        full_content_list: List[str] = []

        if num_pages == 0:
            # Non-paginated documents (DOCX)
            content, temp_file = self._process_linear(doc, file_output_dir, file_cache_dir)
            full_content_list.append(content)
            temp_files_to_delete.append(temp_file)
        else:
            # Paginated documents (PDF, PPTX, XLSX)
            contents, temp_files = self._process_paginated(doc, num_pages, file_output_dir, file_cache_dir)
            full_content_list.extend(contents)
            temp_files_to_delete.extend(temp_files)

        final_markdown_path = file_output_dir / f"{filename}.md"
        markdown_content = "".join(full_content_list).strip()

        with open(final_markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Cleanup temporary files
        self._cleanup_artifacts(file_cache_dir, file_output_dir, temp_files_to_delete)
        # TODO:
        elapsed_time = time.time() - start_time
        print(f"Parser execution time: {elapsed_time:.2f} seconds")
        return {
            "format": path_obj.suffix,
            "markdown": markdown_content,
            "output_dir": str(file_output_dir),
            "num_pages": num_pages
        }

    def _process_linear(
        self, doc, output_dir: Path, cache_dir: Path
    ) -> Tuple[str, Path]:
        """Process non-paginated documents. Images saved to output_dir/images/."""
        temp_filename = cache_dir / "_temp_full.md"

        content = self._save_and_read_temp(
            doc=doc,
            temp_path=temp_filename,
            artifacts_dir=Path("images"),
            page_no=None
        )
        return content, temp_filename

    def _process_paginated(
        self, doc, num_pages: int, output_dir: Path, cache_dir: Path
    ) -> Tuple[List[str], List[Path]]:
        """Process paginated documents. Images saved to output_dir/page_xxxx/."""
        contents = []
        temp_files = []

        for page_num in range(1, num_pages + 1):
            page_folder_name = f"page_{page_num:04d}"
            temp_filename = cache_dir / f"_temp_{page_folder_name}.md"

            page_text = self._save_and_read_temp(
                doc=doc,
                temp_path=temp_filename,
                artifacts_dir=Path(page_folder_name),
                page_no=page_num
            )

            header = f"\n\n- Page {page_num} -\n\n"
            contents.append(header + page_text)
            temp_files.append(temp_filename)

        return contents, temp_files

    def _save_and_read_temp(
        self, doc, temp_path: Path, artifacts_dir: Path, page_no: Optional[int]
    ) -> str:
        """Save document to temp markdown and return content."""
        doc.save_as_markdown(
            filename=temp_path,
            page_no=page_no,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=artifacts_dir
        )
        with open(temp_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _cleanup_artifacts(self, cache_dir: Path, output_dir: Path, temp_files: List[Path]):
        """Remove temporary markdown files and empty image directories."""
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

        if output_dir.exists():
            for dir_path in output_dir.iterdir():
                if dir_path.is_dir() and (dir_path.name == "images" or dir_path.name.startswith("page_")):
                    has_files = any(item.is_file() for item in dir_path.rglob('*'))
                    if not has_files:
                        shutil.rmtree(dir_path, ignore_errors=True)

class DocumentProcessor:
    """
    Document processor using Docling parser.
    Converts multiple documents to Markdown format.
    """
    def __init__(self, output_base_dir: Optional[str] = None):
        self._docling_parser = DoclingParser(output_base_dir=output_base_dir)

    def process_file(self, display_name: str, file_obj: BytesIO) -> Dict[str, Any]:
        """
        Process single document to Markdown.

        Args:
            display_name: filename with ext
            file_obj: File object

        Returns:
            Dict with 'format', 'markdown', 'output_dir', 'num_pages'
        """
        return self._docling_parser.parse(display_name, file_obj)

    def process(self, file_dict: Dict[str, BytesIO]) -> Dict[str, str]:
        """
        Process multiple documents to Markdown.
        Args:
            file_dict: Dict mapping filenames to BytesIO objects

        Returns:
            Dict mapping filenames to Markdown text
        """
        results: dict[str, str] = {}
        for display_name, file_obj in file_dict.items():
            parse_result = self._docling_parser.parse(display_name, file_obj)
            results[display_name] = parse_result['markdown']

        return results


if __name__ == "__main__":
    folder = Path("/data/yunju/pdfs")

    file_list = [p.resolve() for p in folder.iterdir() if p.is_file()]
    print("Found files:", file_list)

    processor = DocumentProcessor(output_base_dir="docling_output_ver1")

    file_dict = {}

    for file_path in file_list:
        with open(file_path, "rb") as f:
                file_dict[file_path.name] = BytesIO(f.read())

    print(f"\nProcessing {len(file_dict)} file(s)...")
    results = processor.process(file_dict)

    for filename, markdown in results.items():
        print(f"\n{filename}:")
        print(f"  Markdown length: {len(markdown)} chars")
        print(f"  Preview: {markdown[:100]}...")
