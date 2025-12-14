# import pikepdf
# def repair_pdf(input_path, output_path):
#     try:
#         # pdf를 열어서 다시 저장하는 것만으로도 XRef 테이블 오류 등이 수정됨
#         with pikepdf.open(input_path, allow_overwriting_input=True) as pdf:
#             pdf.save(output_path)
#         print(f"Repaired: {output_path}")
#     except Exception as e:
#         print(f"Error repairing PDF: {e}")

# repair_pdf("/home/shaush/work/sample/25.8._._0807.1.pdf", "fixed.pdf")

# import io
# io.BytesIO()

import subprocess

def repair_pdf_ghostscript(input_path, output_path):
    # Ghostscript 명령어 구성
    # -o: 출력 파일 지정
    # -sDEVICE=pdfwrite: PDF를 다시 씀
    # -dPDFSETTINGS=/default: 표준 설정 사용
    # -dNOPAUSE -dBATCH -dQUIET: 배치 처리용 플래그
    cmd = [
        "gs",
        "-o", output_path,
        "-sDEVICE=pdfwrite",
        "-dPDFSETTINGS=/default",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        input_path
    ]
    
    try:
        print("Ghostscript로 PDF 구조 재조립 중...")
        subprocess.run(cmd, check=True)
        print(f"성공! 재조립된 파일: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Ghostscript 실행 실패: {e}")
    except FileNotFoundError:
        print("에러: 시스템에 'gs' (Ghostscript)가 설치되어 있지 않습니다.")

# 실행
repair_pdf_ghostscript("/home/shaush/pdfs/25.8._._0807.1.pdf", "/sample/fixed_25.8._._0807s.pdf")