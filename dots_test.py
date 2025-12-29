# parse_pdf_with_dotsocr_to_md.py
import argparse
import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps for image export/cropping
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None


@dataclass
class ImagePlaceholder:
    mode: str  # "referenced" | "embedded" | "placeholder"
    images_dirname: str = "images"


def has_korean_text_quick(pdf_path: Path, max_pages: int = 2) -> bool:
    """Heuristic: if PDF has selectable text containing Hangul in first few pages.
    If it's scanned-only, this will likely return False.
    """
    if fitz is None:
        return False
    doc = fitz.open(str(pdf_path))
    n = min(max_pages, doc.page_count)
    for i in range(n):
        t = doc.load_page(i).get_text("text") or ""
        if any("\uAC00" <= ch <= "\uD7A3" for ch in t):
            doc.close()
            return True
    doc.close()
    return False


def run_dotsocr_parser(
    input_pdf: Path,
    out_dir: Path,
    prompt_mode: str,
    num_thread: int,
    use_hf: bool = False,
) -> Tuple[Path, Path]:
    """
    Runs the official parser:
      python3 dots_ocr/parser.py <pdf> --num_thread N --prompt <prompt_mode> [--use_hf true]
    README says it writes:
      - <stem>.json (layout cells with bbox/category/text)
      - <stem>.md   (concatenated markdown)
    :contentReference[oaicite:3]{index=3}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "dots_ocr/parser.py",
        str(input_pdf),
        "--num_thread",
        str(num_thread),
        "--prompt",
        prompt_mode,
    ]
    if use_hf:
        cmd += ["--use_hf", "true"]

    # Run inside repo root; parser typically writes outputs next to input,
    # so we copy results into out_dir after.
    subprocess.run(cmd, check=True)

    stem = input_pdf.stem
    json_path = input_pdf.with_suffix(".json")
    md_path = input_pdf.with_suffix(".md")

    if not json_path.exists():
        raise FileNotFoundError(f"Expected dots.ocr JSON output not found: {json_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"Expected dots.ocr MD output not found: {md_path}")

    # Move/copy into out_dir to keep outputs together
    out_json = out_dir / f"{stem}.json"
    out_md = out_dir / f"{stem}.md"
    out_json.write_bytes(json_path.read_bytes())
    out_md.write_bytes(md_path.read_bytes())

    return out_json, out_md


def normalize_bbox(b: Any) -> Optional[Tuple[int, int, int, int]]:
    """Accepts bbox formats like [x1,y1,x2,y2] (numbers)."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        return x1, y1, x2, y2
    except Exception:
        return None


def guess_page_key(cell: Dict[str, Any]) -> Optional[int]:
    """Try common page index keys."""
    for k in ("page", "page_id", "page_num", "page_index"):
        if k in cell:
            try:
                return int(cell[k])
            except Exception:
                pass
    return None


def render_pdf_pages(pdf_path: Path, dpi: int = 200) -> List[Path]:
    """
    Render each page to an image (PNG) with PyMuPDF.
    Note: dots.ocr README mentions a preprocess that upsamples images to dpi 200 in their eval pipeline. :contentReference[oaicite:4]{index=4}
    We use 200 dpi to align bbox->pixel mapping as much as possible.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for image export. pip install pymupdf")
    if Image is None:
        raise RuntimeError("Pillow is required for image cropping. pip install pillow")

    doc = fitz.open(str(pdf_path))
    out: List[Path] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    tmp_dir = pdf_path.parent / f".__render_{pdf_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for p in range(doc.page_count):
        page = doc.load_page(p)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = tmp_dir / f"page_{p:04d}.png"
        pix.save(str(img_path))
        out.append(img_path)

    doc.close()
    return out


def image_to_data_uri(img_path: Path) -> str:
    b = img_path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_markdown_from_cells(
    pdf_path: Path,
    cells: List[Dict[str, Any]],
    image_ph: ImagePlaceholder,
    add_page_markers: bool,
    dpi: int = 200,
) -> str:
    """
    Build a markdown by iterating cells in reading order (as returned by dots.ocr parser).
    We insert:
      - Picture -> ![image](path) or embedded data URI
      - Formula -> $$ latex $$
      - Table -> keep HTML as-is
      - Others -> keep markdown text as-is
    This matches the README prompt rules: Formula=LaTeX, Table=HTML, Others=Markdown. :contentReference[oaicite:5]{index=5}
    """
    out_lines: List[str] = []

    # Prepare page renders if we need to export/crop images
    need_images = image_ph.mode in ("referenced", "embedded")
    rendered_pages: List[Path] = []
    if need_images:
        rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)

    # Ensure images dir
    images_dir = pdf_path.parent / image_ph.images_dirname
    if need_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    current_page: Optional[int] = None
    picture_counter: Dict[int, int] = {}

    for cell in cells:
        cat = str(cell.get("category") or cell.get("type") or "").strip()
        text = cell.get("text")
        if text is None:
            # some outputs may use "content"
            text = cell.get("content", "")
        text = str(text)

        page_idx = guess_page_key(cell)
        # If not present, we can't page-split reliably.
        if add_page_markers and page_idx is not None and page_idx != current_page:
            current_page = page_idx
            out_lines.append(f"\n\n<!-- PAGE {current_page + 1} -->\n")

        if cat.lower() == "picture":
            if image_ph.mode == "placeholder":
                out_lines.append("\n\n<!-- image -->\n")
                continue

            # referenced / embedded
            bbox = normalize_bbox(cell.get("bbox"))
            if bbox is None or page_idx is None or page_idx >= len(rendered_pages):
                # Fallback if bbox/page not present
                out_lines.append("\n\n<!-- image (bbox/page missing) -->\n")
                continue

            # Crop picture from rendered page image
            page_img_path = rendered_pages[page_idx]
            with Image.open(page_img_path) as im:
                x1, y1, x2, y2 = bbox
                crop = im.crop((x1, y1, x2, y2))

                picture_counter.setdefault(page_idx, 0)
                picture_counter[page_idx] += 1
                crop_path = images_dir / f"page_{page_idx:04d}_img_{picture_counter[page_idx]:03d}.png"
                crop.save(crop_path)

            if image_ph.mode == "referenced":
                rel = os.path.relpath(crop_path, start=pdf_path.parent)
                out_lines.append(f"\n\n![image]({rel})\n")
            else:
                data_uri = image_to_data_uri(crop_path)
                out_lines.append(f"\n\n![image]({data_uri})\n")

            continue

        if cat.lower() == "formula":
            # README prompt says formula text should be LaTeX. :contentReference[oaicite:6]{index=6}
            latex = text.strip()
            if not latex:
                continue
            # Avoid double-wrapping if already looks like math
            if latex.startswith("$"):
                out_lines.append("\n" + latex + "\n")
            else:
                out_lines.append("\n\n$$\n" + latex + "\n$$\n")
            continue

        if cat.lower() == "table":
            # README prompt says table text should be HTML. :contentReference[oaicite:7]{index=7}
            html = text.strip()
            if html:
                out_lines.append("\n\n" + html + "\n")
            continue

        # Default: keep markdown-ish text
        t = text.strip()
        if t:
            out_lines.append("\n\n" + t + "\n")

    return "".join(out_lines).strip() + "\n"


def load_cells(json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # dots.ocr output is described as "layout elements (bbox/category/text)". :contentReference[oaicite:8]{index=8}
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    # sometimes it might be {"cells":[...]} etc.
    for k in ("cells", "items", "elements", "layouts", "result"):
        v = data.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    raise ValueError(f"Unrecognized JSON structure in {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--image_mode",
        default="referenced",
        choices=["referenced", "embedded", "placeholder"],
        help="How to place images in markdown",
    )
    ap.add_argument("--add_page_markers", action="store_true", help="Add <!-- PAGE N --> markers")
    ap.add_argument("--dpi", type=int, default=200, help="PDF render dpi for cropping")
    ap.add_argument("--num_thread", type=int, default=16, help="dots.ocr parser threads for pdf")
    ap.add_argument(
        "--prompt_mode",
        default=None,
        help="dots.ocr prompt mode (e.g., prompt_layout_all_en, prompt_layout_only_en, prompt_ocr)",
    )
    ap.add_argument("--use_hf", action="store_true", help="Use HF inference instead of vLLM server")
    ap.add_argument(
        "--prefer_disable_ocr",
        action="store_true",
        help=(
            "Try to disable recognition by using a detection-only prompt. "
            "WARNING: markdown will be poor/empty if no text is recognized."
        ),
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # OCR toggle interpretation:
    # - dots.ocr is a VLM that does recognition; there isn't a separate 'traditional OCR engine' toggle in README.
    # - You CAN switch prompts: layout_only disables recognition, but then you can't really output useful markdown.
    # :contentReference[oaicite:9]{index=9}
    prompt_mode = args.prompt_mode
    if prompt_mode is None:
        if args.prefer_disable_ocr:
            prompt_mode = "prompt_layout_only_en"
        else:
            prompt_mode = "prompt_layout_all_en"

    # If Korean is likely present and user tried to disable OCR, force full parse prompt
    if args.prefer_disable_ocr:
        if has_korean_text_quick(pdf_path):
            prompt_mode = "prompt_layout_all_en"

    img_ph = ImagePlaceholder(mode=args.image_mode, images_dirname="images")

    t0 = time.time()

    # 1) Run official dots.ocr parser to produce base JSON/MD
    json_path, _md_path = run_dotsocr_parser(
        input_pdf=pdf_path,
        out_dir=out_dir,
        prompt_mode=prompt_mode,
        num_thread=args.num_thread,
        use_hf=args.use_hf,
    )

    # 2) Load cells and rebuild markdown with page markers + image placeholders
    cells = load_cells(json_path)
    rebuilt_md = build_markdown_from_cells(
        pdf_path=pdf_path,
        cells=cells,
        image_ph=img_ph,
        add_page_markers=args.add_page_markers,
        dpi=args.dpi,
    )

    md_out = out_dir / f"{pdf_path.stem}.repacked.md"
    md_out.write_text(rebuilt_md, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[DONE] elapsed_sec={elapsed:.3f}")
    print(f"[OUT]  json={json_path}")
    print(f"[OUT]  md  ={md_out}")


if __name__ == "__main__":
    main()
