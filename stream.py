#  DocTool 클래스와 실행을 스트리밍 방식으로 수정

class DocTool:
    """
    High-level document processing tool with Multi-GPU/CPU support.
    Now supports Streaming output.
    """

    def __init__(
        self,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        chunk_page_size: int = 10,
        worker_restart_interval: int = 20,
        cpu_workers: int = 4,
    ):
        self.config = ParserConfig(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            chunk_page_size=chunk_page_size,
            worker_restart_interval=worker_restart_interval,
            cpu_workers=cpu_workers,
        )

    def stream(self, file_dict: Dict[str, BytesIO]):
        """
        Process multiple documents and yield results as soon as they are ready.
        (Generator function)
        
        Args:
            file_dict: Dictionary mapping filenames to BytesIO objects
            
        Yields:
            Document objects one by one.
        """
        num_gpus = torch.cuda.device_count()
        print(f"[DocTool] Detected {num_gpus} GPU(s)")

        # 1. Worker 설정
        if num_gpus > 0:
            num_workers = num_gpus
            gpu_ids = list(range(num_gpus))
            cpus_per_worker = None
        else:
            num_workers = self.config.cpu_workers
            gpu_ids = None
            try:
                allowed_cpus = sorted(os.sched_getaffinity(0))
                total_cpus = len(allowed_cpus)
                cpus_per_worker = max(1, total_cpus // num_workers)
            except Exception:
                cpus_per_worker = None

        # 2. 작업 준비 (Task Preparation)
        all_tasks = []
        
        # 파일별 청크 개수를 추적하기 위한 메타데이터
        # key: original_filename, value: { "total": int, "collected": list[ChunkResult] }
        doc_buffers = {} 
        
        # 남은 처리 대상 파일 수
        pending_files_count = 0

        for filename, file_stream in file_dict.items():
            ext = Path(filename).suffix.lower()
            file_stream.seek(0)
            file_bytes = file_stream.read()

            if ext == '.pdf':
                # PDF는 청크로 분할
                chunks = _split_pdf_to_chunks(
                    file_id=filename,
                    pdf_bytes=file_bytes,
                    chunk_page_size=self.config.chunk_page_size
                )
                
                doc_buffers[filename] = {
                    "total": len(chunks),
                    "collected": [],
                    "start_time": time.perf_counter()
                }
                
                for chunk_filename, chunk_index, chunk_stream, start_page in chunks:
                    chunk_stream.seek(0)
                    chunk_bytes = chunk_stream.read()
                    all_tasks.append((
                        chunk_filename,
                        chunk_index,
                        filename,    # original_file_id
                        chunk_bytes,
                        file_bytes,  # needed for chart extraction context if applicable
                        start_page   # page_offset
                    ))
            else:
                # PDF가 아닌 파일은 통째로 1개의 청크처럼 처리
                # Worker가 확장자를 보고 알아서 처리하도록 함
                doc_buffers[filename] = {
                    "total": 1,
                    "collected": [],
                    "start_time": time.perf_counter()
                }
                
                all_tasks.append((
                    filename,       # chunk_filename (same as original)
                    0,              # chunk_index
                    filename,       # original_file_id
                    file_bytes,     # chunk_bytes (entire file)
                    file_bytes,     # file_bytes
                    0               # page_offset
                ))

            pending_files_count += 1

        total_tasks = len(all_tasks)
        print(f"[DocTool] Total files: {len(file_dict)}, Total tasks (chunks): {total_tasks}")

        # 3. Manager 및 Worker 시작
        config_dict = self.config.__dict__.copy() # dataclass to dict
        
        manager = WorkerManager(
            num_workers=num_workers,
            gpu_ids=gpu_ids,
            config_dict=config_dict,
            worker_restart_interval=self.config.worker_restart_interval,
            cpus_per_worker=cpus_per_worker
        )
        manager.start_workers()

        # 4. 작업 큐에 넣기
        for task in all_tasks:
            manager.task_queue.put(task)

        # 5. 결과 수집 및 스트리밍 (Streaming Loop)
        received_chunks = 0
        
        try:
            while pending_files_count > 0:
                # 워커 생존 확인 및 재시작
                for i in range(manager.num_workers):
                    if not manager.processes[i].is_alive():
                        # 워커가 죽었으면 재시작 (작업이 다 끝나기 전이라면)
                        # 주의: 작업 큐에 남아있는 작업은 살아있는 워커가 가져가지만, 
                        # 죽은 워커가 처리 중이던 작업은 유실될 수 있음 (여기선 단순 재시작만 구현)
                        gpu_id = manager.gpu_ids[i] if manager.gpu_ids else None
                        print(f"🔄 [Manager] Worker died, restarting...")
                        manager.restart_worker(i, gpu_id)

                try:
                    # 결과 대기
                    chunk_result = manager.result_queue.get(timeout=5)
                    received_chunks += 1
                    
                    fid = chunk_result.original_file_id
                    buffer = doc_buffers[fid]
                    
                    # 결과 버퍼에 추가
                    buffer["collected"].append(chunk_result)
                    
                    # 해당 파일의 모든 청크가 모였는지 확인
                    if len(buffer["collected"]) == buffer["total"]:
                        # 1. 병합 (Merge)
                        merged_doc = self._merge_single_file(fid, buffer["collected"])
                        
                        # 2. 수행 시간 계산 및 로그
                        elapsed = time.perf_counter() - buffer["start_time"]
                        print(f"✅ [Yield] {fid} ready ({elapsed:.2f}s)")
                        
                        # 3. 결과 반환 (Yield)
                        yield merged_doc
                        
                        # 4. 메모리 정리 (버퍼 삭제)
                        del doc_buffers[fid]
                        pending_files_count -= 1
                        
                except Empty:
                    # 타임아웃 발생 시, 아직 작업이 남았는데 모든 워커가 죽었는지 체크
                    alive_workers = sum(1 for p in manager.processes if p.is_alive())
                    if alive_workers == 0 and pending_files_count > 0:
                        print("[DocTool] Critical: All workers are dead but tasks remain.")
                        break
                    continue
                except Exception as e:
                    print(f"[DocTool] Error in streaming loop: {e}")
                    traceback.print_exc()
                    break

        finally:
            # 6. 종료 처리
            manager.shutdown()
            print("[DocTool] Streaming finished.")

    def _merge_single_file(self, file_id: str, chunks: List[ChunkResult]) -> Document:
        """Merge chunks for a single file into a Document object."""
        # 청크 인덱스 순 정렬
        chunks.sort(key=lambda x: x.chunk_index)
        
        # 에러 체크
        failed_chunks = [c for c in chunks if not c.success]
        if failed_chunks:
            print(f"[Warning] {file_id} has {len(failed_chunks)} failed chunks.")

        # 텍스트 병합
        merged_text = "\n\n".join([c.text for c in chunks if c.success])

        # 이미지 병합
        all_images = []
        for chunk in chunks:
            if chunk.success and chunk.images:
                all_images.extend(chunk.images)

        return Document(
            id=file_id,
            text=merged_text,
            images=all_images if all_images else None
        )


if __name__ == "__main__":
    # Windows/CUDA multiprocessing fix
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    input_folder = Path("/home/shaush/pdfs") # 경로 수정 필요
    output_root = Path("/home/shaush/work/parsed-outputs")
    log_file_path = output_root / "parsing_log.txt"

    output_root.mkdir(parents=True, exist_ok=True)
    
    # 입력 파일 읽기
    file_list = [p.resolve() for p in input_folder.iterdir() if p.is_file()]
    print(f"Found {len(file_list)} files.")
    
    file_dict = {}
    for file_path in file_list:
        with open(file_path, "rb") as f:
            file_dict[file_path.name] = BytesIO(f.read())

    # 프로세서 초기화
    processor = DocTool(
        chunk_page_size=10,
        worker_restart_interval=20,
        cpu_workers=2
    )

    print("Starting Streaming Process...")
    start_time = time.perf_counter()

    # Log 파일 열기 (Append 모드 혹은 Write 모드)
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Streaming Processing Started at {time.ctime()}\n")
        
        # stream() 제너레이터 순회
        count = 0
        for doc in processor.stream(file_dict):
            count += 1
            filename = doc.id
            save_path = output_root / (Path(filename).stem + ".md")

            # 결과 저장
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(doc.text)
                
                # 로그 기록
                num_images = len(doc.images) if doc.images else 0
                log_msg = f"[{count}] Saved {filename} | Images: {num_images}"
                print(log_msg)
                
                log_file.write(log_msg + "\n")
                log_file.write(f"   - Text len: {len(doc.text)}\n")
                log_file.flush() # 즉시 파일에 쓰기

            except Exception as e:
                print(f"Failed to save {filename}: {e}")

    total_time = time.perf_counter() - start_time
    print(f"All done! Total time: {total_time:.2f}s")