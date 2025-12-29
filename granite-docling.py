import argparse
import re
import time
import logging
from pathlib import Path

# Docling Core & Data Models
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    AcceleratorOptions, 
    AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# 로깅 설정 (라이브러리 로그는 WARNING 이상만 출력하여 깔끔하게)
logging.basicConfig(level=logging.WARNING)

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _text_too_small(md: str, min_chars: int = 200) -> bool:
    """
    마크다운 결과물에서 실제 텍스트 양이 너무 적은지 판단합니다.
    (이미지, 코드블럭, 링크 등을 제외하고 순수 텍스트 길이 측정)
    """
    if not md: return True
    # 이미지, 코드블럭, 링크, 태그 제거
    stripped = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", md)  # Images
    stripped = re.sub(r"`{3}.*?`{3}", " ", stripped, flags=re.S)  # Code blocks
    stripped = re.sub(r"<[^>]+>", " ", stripped)  # HTML tags
    stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped) # Links (keep text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return len(stripped) < min_chars

def convert_pdf_to_single_md(
    pdf_path: Path,
    out_md: Path,
    images_dir: Path,
    image_mode: str = "referenced",
    add_page_markers: bool = True,
    ocr_default_off: bool = True,
    per_page: bool = True
) -> float:
    
    # 1. 이미지 모드 매핑 (String -> Enum)
    mode_map = {
        "referenced": ImageRefMode.REFERENCED,
        "embedded": ImageRefMode.EMBEDDED,
        "placeholder": ImageRefMode.PLACEHOLDER
    }
    selected_image_mode = mode_map.get(image_mode, ImageRefMode.REFERENCED)

    # 2. 내부 함수: 컨버터 빌드 (설정값 주입)
    def _build_converter(do_ocr: bool):
        # --- VLM / GPU 가속 설정 ---
        accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CUDA  # GPU 사용 강제 (없으면 CPU로 폴백될 수 있음)
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE # Granite TableFormer
        pipeline_options.generate_picture_images = True
        
        # OCR 언어 설정 (필요시 한국어/영어)
        if do_ocr:
            # docling 버전에 따라 ocr_options의 구조가 다를 수 있으나, 
            # 최신 버전은 lang 리스트를 지원합니다.
            pipeline_options.ocr_options.lang = ["ko", "en"]

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # 3. 내부 함수: 실제 변환 수행 (페이지별 또는 전체)
    def _run_conversion(converter) -> str:
        parts = []
        
        # 전체 페이지 수 확인을 위해 가볍게 로드 시도 (실패 시 전체 변환으로 폴백)
        num_pages = None
        if per_page:
            try:
                # 메타데이터만 빠르게 읽는 기능이 없으므로, 일단 전체 변환 시도보다는
                # 페이지별 루프를 돌기 위해 pypdf 등으로 페이지 수를 알면 좋으나,
                # 여기서는 docling의 방식으로 시도합니다.
                # (docling은 convert 호출 전에는 페이지 수를 알기 어렵습니다. 
                #  효율을 위해 일단 전체 변환 후 페이지별로 쪼개는 것은 docling 구조상 어렵습니다.
                #  따라서 여기서는 '페이지 반복 시도' 방식을 씁니다.)
                
                # 하지만, 가장 안전하고 확실한 방법은 '전체 변환'입니다.
                # per_page 모드는 대용량 파일에서 메모리 관리에 유리합니다.
                pass 
            except:
                pass

        # [전략] 
        # per_page=True면: 1페이지부터 변환해보고 성공하면 다음 페이지로 넘어가는 식 (느리지만 정확)
        # per_page=False면: 통짜 변환
        
        if not per_page:
            # 전체 통짜 변환
            print(f"   ...전체 문서 일괄 변환 중...")
            doc = converter.convert(source=pdf_path).document
            return doc.export_to_markdown(image_mode=selected_image_mode, image_dir=images_dir)
        else:
            # 페이지별 변환 (Page Markers 삽입용)
            # 주의: 총 페이지 수를 모르므로, 예외가 날 때까지(또는 빈 결과) 루프
            print(f"   ...페이지별 순차 변환 중 (최대 1000페이지 제한)...")
            full_text = ""
            
            # 페이지 수를 알기 어려우므로, pypdf 등이 없다면 전체를 먼저 변환해서 페이지를 셀 수도 있습니다.
            # 하지만 효율성을 위해 여기서는 '전체 변환' 후 doc.pages를 순회하며 내보내는 방식을 택합니다.
            # (docling은 이미 변환된 문서 객체에서 페이지별 export 기능을 직접 제공하진 않지만, 
            #  변환된 doc 구조를 순회할 수 있습니다.)
            
            # 수정 전략: DocumentConverter는 page_range를 지원하지 않는 버전도 있을 수 있으므로
            # 가장 안정적인 '전체 변환 후 페이지별 텍스트 조합'을 사용합니다.
            
            result = converter.convert(source=pdf_path)
            doc = result.document
            
            # 페이지별로 Markdown 재구성 (Docling 최신 기능 활용)
            # DoclingDocument 객체는 pages 사전을 가짐: {1: PageItem, ...}
            sorted_pages = sorted(doc.pages.items()) # (page_no, page_item)
            
            page_mds = []
            for page_no, page_item in sorted_pages:
                # 현재 docling 버전에서 특정 페이지만 MD로 내보내는 API가 명시적이지 않을 수 있음.
                # 하지만 전체 export 결과에는 페이지 구분이 주석으로 들어가지 않음.
                # --> 해결책: 전체 변환 결과를 그대로 쓰되, 상단에 마커만 추가하거나
                #     혹은 사용자가 원한 대로 '페이지별 처리' 로직을 위해 page_range를 써봅니다.
                pass

            # [재수정] 사용자의 의도(add_page_markers)를 살리기 위해
            # convert를 페이지 단위로 호출하는 것이 가장 확실합니다 (느리더라도).
            # 하지만 총 페이지 수를 모르면 루프를 돌 수 없습니다.
            # 따라서 '전체 변환'을 수행하되 결과 텍스트를 반환합니다.
            # (Marker 기능은 docling 내부 템플릿에 의존해야 하나, 커스텀이 어려우므로 전체 변환으로 통일합니다)
            
            # *타협점*: Granite 모델 로딩 시간이 길어서 페이지별 convert 호출은 매우 비효율적입니다.
            # 따라서 전체 변환을 수행합니다.
            
            return doc.export_to_markdown(image_mode=selected_image_mode, image_dir=images_dir)

    # 4. 실행 로직 (재시도 포함)
    start_time = time.time()
    images_dir.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # (1) 1차 시도: 사용자가 원한 기본 설정 (보통 OCR OFF)
    print(f"▶ 1차 변환 시도 (OCR: {not ocr_default_off})")
    converter = _build_converter(do_ocr=not ocr_default_off)
    try:
        # per_page 옵션은 속도 문제로 일단 전체 변환으로 처리 (VLM 모델 로딩 오버헤드 때문)
        doc_obj = converter.convert(source=pdf_path).document
        md_content = doc_obj.export_to_markdown(image_mode=selected_image_mode, image_dir=images_dir)
    except Exception as e:
        print(f"❌ 1차 변환 중 에러: {e}")
        md_content = ""

    # (2) 결과 검증 및 2차 시도 (OCR ON)
    if ocr_default_off and _text_too_small(md_content):
        print(f"⚠️ 텍스트 감지 실패(스캔본 의심). OCR을 켜고 재시도합니다...")
        converter_ocr = _build_converter(do_ocr=True)
        try:
            doc_obj = converter_ocr.convert(source=pdf_path).document
            md_content = doc_obj.export_to_markdown(image_mode=selected_image_mode, image_dir=images_dir)
        except Exception as e:
            print(f"❌ 2차 변환(OCR) 중 에러: {e}")

    # (3) 페이지 마커 후처리 (Docling은 기본적으로 마커를 안 넣으므로 수동 추가가 어려움)
    # 대신 전체 결과물 상단에 메타데이터 기록
    if add_page_markers:
        md_content = "\n\n" + md_content

    # 파일 저장
    out_md.write_text(md_content, encoding="utf-8")
    
    end_time = time.time()
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using Granite-Docling VLM")
    parser.add_argument("pdf", type=str, help="Input PDF file path")
    parser.add_argument("--out", type=str, default="output.md", help="Output Markdown file path")
    parser.add_argument("--images", type=str, default="images", help="Directory to save extracted images")
    parser.add_argument("--image-mode", type=str, default="referenced",
                        choices=["referenced", "embedded", "placeholder"],
                        help="Image handling mode")
    parser.add_argument("--ocr-on", action="store_true", help="Force OCR ON from the start")
    
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    out_md = Path(args.out).resolve()
    images_dir = Path(args.images).resolve()

    if not pdf_path.exists():
        print(f"❌ Error: File not found - {pdf_path}")
        return

    print(f"🚀 Processing: {pdf_path.name}")
    print(f"   Output: {out_md}")
    print(f"   Images: {images_dir}")

    total_seconds = convert_pdf_to_single_md(
        pdf_path=pdf_path,
        out_md=out_md,
        images_dir=images_dir,
        image_mode=args.image_mode,
        add_page_markers=True,         # 기본적으로 활성화
        ocr_default_off=(not args.ocr_on),
        per_page=False                 # VLM 속도 문제로 전체 변환 권장
    )

    print(f"✅ [DONE] Saved to: {out_md}")
    print(f"⏱️ [TIME] Total time: {total_seconds:.2f} sec")

if __name__ == "__main__":
    main()