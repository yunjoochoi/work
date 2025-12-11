import os
import sys
from typing import Literal, Dict, Any, Optional, List, Tuple
from pathlib import Path
from io import BytesIO
import shutil
import tempfile
import fitz
import time

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, ThreadedPdfPipelineOptions
from docling_core.types.doc import DocItem, PictureItem
from docling_core.types.io import DocumentStream
from docling_core.types.doc.base import ImageRefMode
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

class DoclingParser:
    """
    Parse documents to Markdown using Docling.
    Supports PDF, DOCX, PPTX, XLSX with page-wise processing and image extraction.
    """
    def __init__(self, output_base_dir: Optional[str] = None):
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.generate_picture_images = False
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
        
        # 1. 경로 및 스트림 설정
        path_obj = Path(display_name)
        filename = path_obj.stem
        
        base_dir = Path(output_dir) if output_dir else self.output_base_dir
        file_output_dir = base_dir / filename
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fitz용으로 bytes 데이터 백업 (I/O Closed 방지)
        file_obj.seek(0)
        file_bytes = file_obj.read()
        
        doc_stream = DocumentStream(name=display_name, stream=BytesIO(file_bytes))

        # 2. Docling 변환 (텍스트 & 표 구조 분석)
        start = time.time()
        result = self.converter.convert(doc_stream)
        doc = result.document
        
        # 3. 하이브리드 마크다운 생성
        final_markdown_path = file_output_dir / f"{filename}.md"
        
        # PDF인 경우: Fitz + Docling 하이브리드 처리
        if path_obj.suffix.lower() == '.pdf':
            self._process_hybrid_pdf(
                doc=doc,
                file_bytes=file_bytes,
                output_dir=file_output_dir,
                markdown_path=final_markdown_path
            )
        # 기타 포맷 (DOCX 등): 기존 방식 (이미지 추출 X, 텍스트만)
        else:
            # DOCX 등은 Linear 처리 (필요시 기존 로직 복구 가능)
            with open(final_markdown_path, "w", encoding="utf-8") as f:
                f.write(doc.export_to_markdown())

        # 4. 정리
        self._cleanup_empty_dirs(file_output_dir)
        
        elapsed = time.time() - start
        print(f"Parser 실행시간: {elapsed:.2f} 초")

        return {
            "format": path_obj.suffix,
            "markdown": str(final_markdown_path),
            "output_dir": str(file_output_dir),
            "num_pages": doc.num_pages()
        }

    def _process_hybrid_pdf(self, doc, file_bytes: bytes, output_dir: Path, markdown_path: Path):
        """
        [Hybrid Logic]
        Fitz의 이미지 위치 정보와 Docling의 '모든 텍스트성 아이템'을 Y좌표 기준으로 섞어서 마크다운을 생성합니다.
        """
        pdf_doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
        
        # 만능 변환기 (Text, Table, List, Formula, Code 등 모든 NodeItem 처리 가능)
        serializer = MarkdownDocSerializer(doc=doc)
        
        with open(markdown_path, "w", encoding="utf-8") as fw:
            
            for page_num in range(1, doc.num_pages() + 1):
                fw.write(f"\n\n## Page {page_num}\n\n")
                
                # 1. 페이지별 폴더 생성
                page_folder = output_dir / f"page_{page_num:04d}"
                page_folder.mkdir(parents=True, exist_ok=True)
                
                # 2. [Fitz] 이미지 추출 (기존과 동일)
                fitz_page = pdf_doc[page_num - 1]
                items_to_sort = [] # 정렬 대상 아이템들 (y_coord, type, content)
                
                img_infos = fitz_page.get_image_info(xrefs=True)
                # ------------------------------------------------------
                for info in img_infos:
                    xref = info["xref"]

                    # PyMuPDF는 벡터 마스크도 XObject처럼 반환하므로 필터링 필요
                    # 실제 bitmap인지 검사
                    try:
                        pix = fitz.Pixmap(pdf_doc, xref)

                        # 진짜 RGB 또는 Gray 비트맵만 추출
                        if pix.n in (1, 3, 4):
                            # crop bbox 정보
                            bbox = info["bbox"]
                            y_coord = bbox[1]   # 상단 y좌표

                            # 저장 경로 (페이지별 폴더 유지)
                            img_name = f"img_{xref}.png"
                            img_save_path = page_dir / img_name
                            pix.save(str(img_save_path))
                            pix = None

                            items_to_sort.append((
                                y_coord,
                                "image",
                                {
                                    "path": f"{page_name}/{img_name}",
                                    "bbox": bbox,
                                    "xref": xref,
                                    "source": "bitmap",
                                }
                            ))

                    except Exception:
                        # 비트맵이 아니면 넘어감 (mask-only or vector)
                        pass


                # 2-2. (B) 벡터 기반 Path 객체 탐지 (그래프/표 선 등)
                #       → 필요하면 clip raster image로 추출
                # rawdict 분석
                raw = fitz_page.get_text("rawdict")

                for block in raw["blocks"]:
                    if block["type"] == 2:  # path → vector shape (chart, box 등)
                        bbox = block["bbox"]
                        y_coord = bbox[1]

                        # clip 렌더링 (벡터를 래스터로 변환)
                        rect = fitz.Rect(bbox)
                        pix = fitz_page.get_pixmap(clip=rect, dpi=200)

                        img_name = f"vector_{int(y_coord)}.png"
                        img_save_path = page_dir / img_name
                        pix.save(str(img_save_path))

                        items_to_sort.append((
                            y_coord,
                            "vector",
                            {
                                "path": f"{page_name}/{img_name}",
                                "bbox": bbox,
                                "source": "vector",
                            }
                        ))

                for idx, info in enumerate(img_infos):
                    xref = info['xref']
                    bbox = info['bbox'] # [x0, y0, x1, y1] (Top-Left Origin)
                    
                    # 이미지 저장 로직
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    img_filename = f"img_{idx:02d}.{image_ext}"
                    img_path = page_folder / img_filename
                    
                    if not img_path.exists():
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                    
                    rel_path = f"page_{page_num:04d}/{img_filename}"
                    md_str = f"\n![Image]({rel_path})\n"
                    
                    # 리스트에 추가 (정렬 키: Y0)
                    items_to_sort.append((bbox[1], 'image', md_str))
                # 3. [Docling] "그림 빼고 모든 아이템" 수집
                doc_page = doc.pages[page_num]
                page_height = doc_page.size.height
                
                # iterate_items는 Title, Text, Table, List, Code, Formula 등 모든걸 순회합니다.
                for item, _ in doc.iterate_items(page_no=page_num):
                    
                    # [핵심 변경] "그림(PictureItem)만 아니면 다 가져온다"
                    if isinstance(item, PictureItem):
                        continue
                        
                    # DocItem을 상속받은 모든 요소(Text, Table, KeyValue 등) 처리
                    if isinstance(item, DocItem):
                        try:
                            # Serializer가 아이템 타입에 맞춰 알아서 마크다운으로 변환해줍니다.
                            # TitleItem -> "# 제목"
                            # ListItem -> "- 항목"
                            # TableItem -> "| 표 |..."
                            item_md = serializer.serialize(item=item).text
                        except Exception as e:
                            print(f"변환 경고 (Page {page_num}): {e}")
                            continue
                        
                        if item_md.strip():
                            # 좌표 변환 (Docling Bottom-Left -> Fitz Top-Left)
                            y0 = 0
                            if item.prov:
                                bbox = item.prov[0].bbox
                                y0 = page_height - bbox.t 
                            else:
                                y0 = 99999 # 위치 정보 없으면 맨 뒤로
                                
                            items_to_sort.append((y0, 'text', item_md))
                # ------------------------------------------------------

                # 4. [Merge] Y좌표(위->아래) 순서로 정렬
                items_to_sort.sort(key=lambda x: x[0])
                
                # 5. 파일 쓰기
                for _, _, content in items_to_sort:
                    fw.write(content)
                    fw.write("\n\n") 

        pdf_doc.close()

    def _cleanup_empty_dirs(self, output_dir: Path):
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
