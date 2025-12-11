import json
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import time
from docling_core.types.doc.base import ImageRefMode
import shutil
def parse_documents(file_paths):
    """
    주어진 파일 경로 리스트를 순회하며 Docling을 사용해 파싱합니다.
    PDF, DOCX, PPTX, XLSX 등의 포맷을 자동으로 감지합니다.
    """
    
    # DocumentConverter 인스턴스 생성
    # 기본 설정으로 대부분의 문서를 처리할 수 있습니다.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False           # True: 이미지 PDF에 OCR 수행 (기본값)
    pipeline_options.do_table_structure = True # True: 테이블 구조 분석 수행 (기본값)
    
    pipeline_options.generate_picture_images = True # 문서 내 그림 추출
    pipeline_options.generate_table_images = True   # 문서 내 표를 이미지로 추출

    # 2. DocumentConverter 인스턴스 생성
    # format_options를 통해 PDF 처리 시 위에서 정의한 옵션을 사용하도록 연결합니다.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    parsed_results = {}

    base_output_dir = Path("docling_parsed_output")
    base_output_dir.mkdir(exist_ok=True)
    print(f"총 {len(file_paths)}개의 파일 처리를 시작합니다...\n")

    for file_path in file_paths:
        path_obj = Path(file_path)
        
        print(f"# 파싱 중: {path_obj.name} ({file_path})")

        try:
            start = time.time()

            # 1. 문서 변환 수행
            result = converter.convert(file_path)
            doc = result.document 

            # 2. 기본 저장 경로 설정 (docling_parsed_output -> docling_img_output 으로 변경 등은 함수 밖에서 설정)
            # 여기서는 전달받은 base_output_dir를 기준으로 합니다.
            file_output_dir = base_output_dir / path_obj.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # 최종 마크다운 파일 경로 (파일명/문서.md)
            markdown_file = file_output_dir / f"{path_obj.stem}.md"

            # 3. 페이지별 변환 및 병합
            full_content_list = []
            temp_files_to_delete = []

            num_pages = doc.num_pages()

            # DOCX, PPTX 등은 페이지 수가 0일 수 있음 -> 전체 문서를 한 번에 처리
            if num_pages == 0:
                # 이미지 폴더명 (상대 경로)
                artifacts_folder = "images"

                # 임시 마크다운 파일 경로
                temp_filename = file_output_dir / "_temp_full.md"
                temp_files_to_delete.append(temp_filename)

                # save_as_markdown 호출 (page_no 없이 전체 문서)
                doc.save_as_markdown(
                    filename=temp_filename,
                    image_mode=ImageRefMode.REFERENCED,
                    artifacts_dir=Path(artifacts_folder)
                )

                # 마크다운 파일 읽기
                with open(temp_filename, 'r', encoding='utf-8') as f:
                    page_text = f.read()

                full_content_list.append(page_text)

            else:
                # PDF 등 페이지가 있는 문서: 페이지별 처리
                for page_num in range(1, num_pages + 1):

                    # 페이지별 이미지 폴더명 (상대 경로)
                    page_folder_name = f"page_{page_num:04d}"

                    # 임시 마크다운 파일 경로
                    temp_filename = file_output_dir / f"_temp_{page_folder_name}.md"
                    temp_files_to_delete.append(temp_filename)

                    # save_as_markdown 호출
                    doc.save_as_markdown(
                        filename=temp_filename,
                        page_no=page_num,
                        image_mode=ImageRefMode.REFERENCED,
                        artifacts_dir=Path(page_folder_name)
                    )

                    # 임시 마크다운 파일 읽기
                    with open(temp_filename, 'r', encoding='utf-8') as f:
                        page_text = f.read()

                    # 페이지 번호 헤더 추가
                    header = f"\n\n- Page {page_num} -\n\n"
                    full_content_list.append(header + page_text)

            # 4. 빈 폴더 정리 (이미지가 없는 폴더 삭제)
            if num_pages == 0:
                # DOCX 등: images 폴더 확인
                images_dir = file_output_dir / "images"
                if images_dir.exists():
                    has_images = False
                    for item in images_dir.rglob('*'):
                        if item.is_file():
                            has_images = True
                            break
                    if not has_images:
                        shutil.rmtree(images_dir, ignore_errors=True)
            else:
                # PDF 등: 페이지별 폴더 확인
                for page_num in range(1, num_pages + 1):
                    page_folder_name = f"page_{page_num:04d}"
                    page_dir = file_output_dir / page_folder_name

                    if page_dir.exists():
                        has_images = False
                        for item in page_dir.rglob('*'):
                            if item.is_file():
                                has_images = True
                                break
                        if not has_images:
                            shutil.rmtree(page_dir, ignore_errors=True)

            # 5. 임시 파일 삭제
            for temp_file in temp_files_to_delete:
                if temp_file.exists():
                    temp_file.unlink()

            # 6. 전체 내용을 하나의 문자열로 합치기
            markdown_content = "".join(full_content_list).strip()

            # 최종 마크다운 파일 저장
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # 7. 결과 딕셔너리에 저장
            parsed_results[path_obj.name] = {
                "format": path_obj.suffix,
                "markdown": markdown_content,
                "output_dir": str(file_output_dir),
            }

            end = time.time()
            elapsed = end - start
            print(f"Parser 실행시간: {elapsed:.2f} 초")
            
            print(f"✅ 완료: {path_obj.name}")
            if num_pages == 0:
                print(f"   - 문서 타입: {path_obj.suffix} (페이지 구분 없음)")
            else:
                print(f"   - 총 페이지 수: {num_pages}")
            print(f"   - Markdown 길이: {len(markdown_content)} 자")
            print(f"   - 저장 위치: {file_output_dir}")

            # 미리보기
            preview = markdown_content[:200].replace('\n', ' ')
            print(f"   - 미리보기: {preview}...\n")

        except Exception as e:
            print(f"❌ 에러 발생 ({path_obj.name}): {str(e)}\n")

    return parsed_results

# --- 실행 예시 ---
if __name__ == "__main__":
    # 테스트할 파일 경로 리스트 (실제 파일 경로로 변경해주세요)
    # docling은 확장자를 보고 적절한 백엔드를 자동으로 선택합니다.
    target_files = [
        "/data/yunju/pdfs/251204_lecture8_llm_agent.pdf",
        "/data/yunju/pdfs/20251128_company_828658000.pdf",
        "/data/yunju/pdfs/not3.pdf",       # PDF
        "/data/yunju/pdfs/과제4_11주차_2025311605_최윤주.docx",    # Word
        "/data/yunju/pdfs/AI_Safety_진단_툴_사례분석.pptx",# PowerPoint
        "/data/yunju/pdfs/tl계열열린재정자료_.xlsx"         # Excel (주의: Docling 버전에 따라 지원 범위가 다를 수 있음)
    ]

    # 파싱 함수 실행
    results = parse_documents(target_files)

    print("\n" + "="*60)
    print("처리 완료!")
    print("="*60)
    for filename, data in results.items():
        print(f"파일: {filename}")
        print(f"  출력 폴더: {data['output_dir']}")
        print(f"  마크다운 파일: {data['output_dir']}/{Path(filename).stem}.md")
        file_ext = Path(filename).suffix.lower()
        if file_ext == '.pdf':
            print(f"  이미지 폴더: {data['output_dir']}/page_xxxx/ (이미지가 있는 페이지만)")
        else:
            print(f"  이미지 폴더: {data['output_dir']}/images/ (이미지가 있는 경우)")
        print()