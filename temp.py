import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.io import DocumentStream

# Constants
DEFAULT_OUTPUT_DIR = "parsed_output"
DEFAULT_IMAGES_SCALE = 2.0
DEFAULT_LAYOUT_SIZE = 8
DEFAULT_TABLE_SIZE = 8


class DoclingParser:
    """
    Parse documents to Markdown using Docling. with page-wise processing and image extraction.
    Provides configurable OCR, table structure detection, and image generation.
    """

    def __init__(
        self,
        output_base_dir: Optional[str] = None,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        images_scale: float = DEFAULT_IMAGES_SCALE,
        layout_batch_size: int = DEFAULT_LAYOUT_SIZE,
        table_batch_size: int = DEFAULT_TABLE_SIZE,
    ):
        """
        Initialize the Docling parser.

        Args:
            output_base_dir: Base directory for output files
            do_ocr: Enable OCR processing
            do_table_structure: Enable table structure detection
            images_scale: Scale factor for generated images
            layout_batch_size: Batch size for layout detection
            table_batch_size: Batch size for table structure recognition
        """

        pipeline_options = self._create_pipeline_options(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            images_scale=images_scale,
            layout_batch_size=layout_batch_size, # page batch
            table_batch_size=table_batch_size,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.output_base_dir = Path(output_base_dir or DEFAULT_OUTPUT_DIR)
        self.output_base_dir.mkdir(exist_ok=True)

    @staticmethod
    def _create_pipeline_options(
        do_ocr: bool,
        do_table_structure: bool,
        images_scale: float,
        layout_batch_size: int,
        table_batch_size: int,
    ) -> PdfPipelineOptions:
        """Create and configure PDF pipeline options."""
        options = PdfPipelineOptions()
        options.do_ocr = do_ocr
        options.do_table_structure = do_table_structure
        options.generate_picture_images = True
        options.images_scale = images_scale
        options.layout_batch_size = layout_batch_size
        options.table_batch_size = table_batch_size
        return options

    def parse(
        self,
        display_name: str,
        file_obj: BytesIO,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Parse a document to Markdown format.

        Args:
            display_name: Display name of the file (including extension)
            file_obj: BytesIO object containing the file data
            output_dir: Optional output directory override
            cache_dir: Optional cache directory (reserved for future use)

        Returns:
            Dictionary containing parsing results with keys:
                - format: File extension
                - markdown: Path to generated markdown file
                - output_dir: Output directory path
                - num_pages: Number of pages in the document
        """
        # TODO:
        start_time = time.time()

        path_obj = Path(display_name)
        filename = path_obj.stem

        base_dir = Path(output_dir) if output_dir else self.output_base_dir
        file_output_dir = base_dir / filename
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # Create document stream for Docling
        doc_stream = DocumentStream(name=display_name, stream=file_obj)

        # Convert document
        result = self.converter.convert(doc_stream)
        doc = result.document

        # Save markdown with page separation
        final_markdown_path = file_output_dir / f"{filename}.md"
        self._save_markdown_with_page_separation(
            doc=doc,
            final_md_path=final_markdown_path,
            output_root_dir=file_output_dir,
        )

        # Clean up empty directories
        self._cleanup_empty_dirs(file_output_dir)

        # TODO:
        elapsed_time = time.time() - start_time
        print(f"Parser execution time: {elapsed_time:.2f} seconds")

        return {
            "format": path_obj.suffix,
            "markdown": str(final_markdown_path),
            "output_dir": str(file_output_dir),
            "num_pages": doc.num_pages(),
        }

    def _save_markdown_with_page_separation(
        self, doc, final_md_path: Path, output_root_dir: Path
    ) -> None:
        """
        Save document as Markdown with page-wise image organization.

        For paginated documents (PDF, etc), saves images in separate folders per page.
        For linear documents (DOCX), saves all images in a single folder.

        Args:
            doc: Docling document object
            final_md_path: Path to the output markdown file
            output_root_dir: Root directory for output files
        """
        
        num_pages = doc.num_pages()

        with open(final_md_path, "w", encoding="utf-8") as file_writer:
            if num_pages == 0:
                self._process_linear_document(doc, file_writer, output_root_dir, final_md_path)
            else:
                self._process_paginated_document(
                    doc, file_writer, output_root_dir, final_md_path, num_pages
                )
        


    def _process_linear_document(
        self, doc, file_writer, output_root_dir: Path, final_md_path: Path
    ) -> None:
        """Process documents without pages (DOCX)."""
        images_dir = output_root_dir / "images"

        # Save images and update references
        new_doc = doc._with_pictures_refs(
            image_dir=images_dir, page_no=None, reference_path=final_md_path.parent
        )

        markdown_output = new_doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
        file_writer.write(markdown_output)

    def _process_paginated_document(
        self,
        doc,
        file_writer,
        output_root_dir: Path,
        final_md_path: Path,
        num_pages: int,
    ) -> None:
        """Process paginated documents (PDF)"""
        for page_num in range(1, num_pages + 1):
            page_folder = output_root_dir / f"page_{page_num:04d}"

            # Save images for this page and update references
            page_doc = doc._with_pictures_refs(
                image_dir=page_folder,
                page_no=page_num,
                reference_path=final_md_path.parent,
            )

            # Export page to markdown
            markdown_output = page_doc.export_to_markdown(
                page_no=page_num, image_mode=ImageRefMode.REFERENCED
            )

            # Write page header and content
            page_header = f"\n\n- Page {page_num} -\n\n"
            file_writer.write(page_header)
            file_writer.write(markdown_output.strip())

    def _cleanup_empty_dirs(self, output_dir: Path) -> None:
        """Remove empty image directories."""
        if not output_dir.exists():
            return

        for dir_path in output_dir.iterdir():
            if not dir_path.is_dir():
                continue

            if dir_path.name == "images" or dir_path.name.startswith(
                "page_"
            ):
                has_files = any(item.is_file() for item in dir_path.rglob("*"))
                if not has_files:
                    shutil.rmtree(dir_path, ignore_errors=True)

class DocumentProcessor:
    """
    High-level document processor using Docling parser.

    Provides a simple interface for converting documents to Markdown format.
    Supports batch processing of multiple documents.
    """

    def __init__(
        self,
        output_base_dir: Optional[str] = None,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ):
        """
        Initialize the document processor.

        Args:
            output_base_dir: Base directory for output files
            do_ocr: Enable OCR processing
            do_table_structure: Enable table structure detection
        """
        self._parser = DoclingParser(
            output_base_dir=output_base_dir,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )

    def process_file(self, display_name: str, file_obj: BytesIO) -> Dict[str, Any]:
        """
        Process a single document to Markdown.

        Args:
            display_name: Filename with extension
            file_obj: File object containing document data

        Returns:
            Dictionary with parsing results containing:
                - format: File extension
                - markdown: Path to generated markdown file
                - output_dir: Output directory path
                - num_pages: Number of pages
        """
        return self._parser.parse(display_name, file_obj)

    def process(self, file_dict: Dict[str, BytesIO]) -> Dict[str, str]:
        """
        Process multiple documents to Markdown.

        Args:
            file_dict: Dictionary mapping filenames to BytesIO objects

        Returns:
            Dictionary mapping filenames to markdown file paths
        """
        results: Dict[str, str] = {}
        for display_name, file_obj in file_dict.items():
            parse_result = self._parser.parse(display_name, file_obj)
            results[display_name] = parse_result["markdown"]
        return results


if __name__ == "__main__":
    folder = Path("/data/yunju/pdfs")

    file_list = [p.resolve() for p in folder.iterdir() if p.is_file()]
    print("Found files:", file_list)

    processor = DocumentProcessor(output_base_dir="docling_output")

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


