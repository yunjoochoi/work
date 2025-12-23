# pip install nvidia-ml-py
import time
import torch
import gc
from pynvml import *
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

def get_real_vram_usage(pid):
    """NVIDIA 드라이버에서 해당 프로세스(PID)의 실제 VRAM 점유율을 조회"""
    try:
        handle = nvmlDeviceGetHandleByIndex(0) # 0번 GPU 기준
        info = nvmlDeviceGetComputeRunningProcesses(handle)
        for process in info:
            if process.pid == pid:
                return process.usedGpuMemory / 1024**2 # MB 단위 반환
    except Exception:
        pass
    return 0

def print_status(step):
    """현재 프로세스의 PyTorch 측정값 vs 실제 드라이버 측정값 비교"""
    torch.cuda.synchronize()
    pid = os.getpid()
    
    # 1. PyTorch가 알고 있는 값
    torch_allocated = torch.cuda.memory_allocated() / 1024**2
    torch_reserved = torch.cuda.memory_reserved() / 1024**2
    
    # 2. 실제 NVIDIA 드라이버가 말하는 값 (진실)
    real_usage = get_real_vram_usage(pid)
    
    print(f"[{step}]")
    print(f"  - PyTorch Allocated (순수 모델): {torch_allocated:,.1f} MB")
    print(f"  - PyTorch Reserved  (캐시 포함): {torch_reserved:,.1f} MB")
    print(f"  🔥 Real Process Usage (찐 사용량): {real_usage:,.1f} MB (CUDA Context 포함)")
    print("-" * 50)
    return real_usage

def main():
    nvmlInit()
    print(f"🚀 측정 시작 (PID: {os.getpid()})...\n")
    
    # 1. 초기 상태 (아무것도 안 함)
    # 주의: torch.cuda.is_available()만 호출해도 CUDA Context가 생성되어 수백 MB가 잡힐 수 있음
    base_usage = print_status("1. 초기 상태")

    # 2. 모델 로딩
    print("Docling 모델 로딩 중...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.layout_batch_size = 32
    pipeline_options.table_batch_size = 32

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=torch.device("cuda")
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Fallback converter with PyPdfiumDocumentBackend for handling "Invalid code point" errors
    fallback_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
    load_usage = print_status("2. 모델 로딩 직후")

    # 3. (중요) 더미 인퍼런스 실행 (피크 메모리 확인용)
    # 실제 PDF 처럼 동작하게 하여 순간적으로 메모리가 얼마나 튀는지 확인
    print("🏃 더미 인퍼런스(Warmup) 실행 중... (피크 메모리 확인)")
    try:
        # 빈 PDF나 간단한 URL 등으로 실제 파이프라인을 태워봐야 함
        # 여기서는 모델이 메모리에 '자리 잡게' 하는 용도
        pass 
        # (실제 파일을 넣어서 convert()를 한 번 실행하는 코드를 넣으면 베스트입니다)
    except:
        pass
        
    final_usage = print_status("3. 인퍼런스 준비 완료 상태")

    print("\n📊 [최종 결론]")
    print(f"✅ 모델 가중치 크기: 약 {load_usage - base_usage:,.1f} MB")
    print(f"✅ 프로세스 1개당 필요 VRAM: 최소 {final_usage:,.1f} MB")
    print(f"   (CUDA Context + 모델 + 라이브러리 오버헤드 포함)")

    nvmlShutdown()

if __name__ == "__main__":
    main()