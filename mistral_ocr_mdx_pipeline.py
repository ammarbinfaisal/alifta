import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pymupdf as fitz  # PyMuPDF


def _data_url(*, mime_type: str, raw_bytes: bytes) -> str:
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _rasterize_pdf_to_image_urls(
    *,
    pdf_path: str,
    dpi: int,
    start_page: int,
    end_page: Optional[int],
    image_format: str,
) -> list[tuple[int, str]]:
    """
    Returns a list of (page_num, image_data_url).
    Pages are 1-based.
    """
    if image_format not in {"png", "jpeg"}:
        raise ValueError("image_format must be 'png' or 'jpeg'")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    start_page = max(1, start_page)
    end_page = total_pages if end_page is None else min(end_page, total_pages)
    if start_page > end_page:
        raise ValueError(f"Invalid page range: start_page={start_page} > end_page={end_page}")

    image_urls: list[tuple[int, str]] = []
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)

    for i in range(start_page - 1, end_page):
        page_num = i + 1
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix)
        if image_format == "png":
            raw = pix.tobytes("png")
            mime = "image/png"
        else:
            raw = pix.tobytes("jpeg")
            mime = "image/jpeg"
        image_urls.append((page_num, _data_url(mime_type=mime, raw_bytes=raw)))

    return image_urls


def create_batch_file(image_urls: list[str], output_file: str) -> None:
    """
    Minimal helper matching the snippet in `annotations.md`/`basic_ocr.md`.

    Prefer `create_batch_file_for_pages` for PDF pipelines where page numbers matter.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for index, url in enumerate(image_urls):
            entry = {
                "custom_id": str(index),
                "body": {
                    "document": {"type": "image_url", "image_url": url},
                    "include_image_base64": True,
                },
            }
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def create_batch_file_for_pages(
    *,
    page_image_urls: list[tuple[int, str]],
    output_file: str,
    include_image_base64: bool = True,
    table_format: Optional[str] = None,
    extract_header: bool = False,
    extract_footer: bool = False,
) -> None:
    with open(output_file, "w", encoding="utf-8") as file:
        for page_num, url in page_image_urls:
            body: dict[str, Any] = {
                "document": {"type": "image_url", "image_url": url},
                "include_image_base64": include_image_base64,
            }
            if table_format is not None:
                body["table_format"] = table_format
            if extract_header:
                body["extract_header"] = True
            if extract_footer:
                body["extract_footer"] = True

            entry = {"custom_id": str(page_num), "body": body}
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_ocr_body(batch_output_line: dict[str, Any]) -> Optional[dict[str, Any]]:
    if "error" in batch_output_line and batch_output_line["error"]:
        return None
    if "response" in batch_output_line and isinstance(batch_output_line["response"], dict):
        resp = batch_output_line["response"]
        if isinstance(resp.get("body"), dict):
            return resp["body"]
        if isinstance(resp.get("data"), dict):
            return resp["data"]
    if isinstance(batch_output_line.get("body"), dict):
        return batch_output_line["body"]
    return None


def _parse_batch_output_jsonl(jsonl_text: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Returns:
      - successes: custom_id -> OCR response body
      - failures: custom_id -> raw error object (best-effort)
    """
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, Any] = {}

    for raw_line in jsonl_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        custom_id = str(obj.get("custom_id", ""))
        body = _extract_ocr_body(obj)
        if body is not None:
            successes[custom_id] = body
        else:
            failures[custom_id] = obj.get("error") or obj

    return successes, failures


@dataclass
class _OcrContext:
    major_heading: Optional[str] = None
    minor_heading: Optional[str] = None
    tail_text: str = ""


def _strip_redundant_leading_headings(mdx: str, prev_context: _OcrContext) -> str:
    lines = mdx.splitlines()
    idx = 0

    def skip_blank_lines(i: int) -> int:
        while i < len(lines) and not lines[i].strip():
            i += 1
        return i

    idx = skip_blank_lines(idx)

    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("# "):
            heading = line[2:].strip()
            if prev_context.major_heading and heading == prev_context.major_heading:
                idx += 1
                idx = skip_blank_lines(idx)
                continue

        if line.startswith("## "):
            heading = line[3:].strip()
            if prev_context.minor_heading and heading == prev_context.minor_heading:
                idx += 1
                idx = skip_blank_lines(idx)
                continue

        break

    return "\n".join(lines[idx:]).lstrip("\n")


def _update_context_from_mdx(mdx: str, context: _OcrContext) -> _OcrContext:
    major = context.major_heading
    minor = context.minor_heading

    for line in mdx.splitlines():
        if line.startswith("# "):
            new_major = line[2:].strip()
            if new_major and new_major != major:
                major = new_major
                minor = None
        elif line.startswith("## "):
            new_minor = line[3:].strip()
            if new_minor:
                minor = new_minor

    tail = " ".join(mdx.split()).strip()
    tail = tail[-800:] if tail else ""
    return _OcrContext(major_heading=major, minor_heading=minor, tail_text=tail)


def _combined_frontmatter(*, volume: int, start_page: int, end_page: int, pdf_name: str) -> str:
    return (
        "---\n"
        f"volume: {volume}\n"
        f"start_page: {start_page}\n"
        f"end_page: {end_page}\n"
        f"source_pdf: {pdf_name}\n"
        "---\n\n"
    )


def _mdx_frontmatter(*, volume: int, page_num: int) -> str:
    return f"---\nvolume: {volume}\npage: {page_num}\n---\n\n"


def _require_mistral_client():
    try:
        from mistralai import Mistral  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'mistralai'. Install with:\n"
            "  uv pip install -p .venv mistralai\n"
            "or update `setup_ocr.sh` to include it."
        ) from e

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Missing env var `MISTRAL_API_KEY`.")
    return Mistral(api_key=api_key)


def _run_batch_pdf_to_mdx(
    *,
    pdf_path: str,
    vol_num: int,
    model: str,
    dpi: int,
    start_page: int,
    end_page: Optional[int],
    image_format: str,
    poll_seconds: float,
    table_format: Optional[str],
    extract_header: bool,
    extract_footer: bool,
    include_image_base64: bool,
    write_combined: bool,
    combined_filename: str,
    keep_batch_files: bool,
) -> None:
    client = _require_mistral_client()

    pdf_name = Path(pdf_path).stem
    output_mdx_dir = Path(f"vols/vol{vol_num}/mdx")
    output_mdx_dir.mkdir(parents=True, exist_ok=True)

    page_image_urls = _rasterize_pdf_to_image_urls(
        pdf_path=pdf_path,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
        image_format=image_format,
    )
    actual_start = page_image_urls[0][0]
    actual_end = page_image_urls[-1][0]

    batch_dir = Path(f"temp_mistral_batch_{pdf_name}_vol{vol_num}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_file = str(batch_dir / "input.jsonl")

    create_batch_file_for_pages(
        page_image_urls=page_image_urls,
        output_file=batch_file,
        include_image_base64=include_image_base64,
        table_format=table_format,
        extract_header=extract_header,
        extract_footer=extract_footer,
    )

    with open(batch_file, "rb") as handle:
        uploaded = client.files.upload(
            file={"file_name": Path(batch_file).name, "content": handle},
            purpose="batch",
        )
    job = client.batch.jobs.create(
        input_files=[uploaded.id],
        model=model,
        endpoint="/v1/ocr",
        metadata={"job_type": "pdf_to_mdx", "pdf": pdf_name, "volume": str(vol_num)},
    )

    while True:
        job = client.batch.jobs.get(job_id=job.id)
        status = getattr(job.status, "value", str(job.status))
        completed = int(job.completed_requests or 0)
        total = int(job.total_requests or 0)
        print(f"[batch] {job.id} status={status} completed={completed}/{total}")

        if status.lower() in {"succeeded", "success", "completed"}:
            break
        if status.lower() in {"failed", "cancelled", "canceled", "expired"}:
            raise SystemExit(f"Batch job ended with status={status}. job_id={job.id}")

        time.sleep(poll_seconds)

    if not job.output_file:
        raise SystemExit(f"Batch job has no output_file. job_id={job.id}")

    output_resp = client.files.download(file_id=job.output_file)
    output_text = output_resp.text

    successes, failures = _parse_batch_output_jsonl(output_text)
    if failures:
        first_keys = ", ".join(list(failures.keys())[:10])
        print(f"[batch] warning: {len(failures)} failures (custom_id: {first_keys})")

    combined_parts: list[str] = []
    context = _OcrContext()

    for page_num, _ in page_image_urls:
        key = str(page_num)
        body = successes.get(key)
        if not body:
            print(f"[mdx] missing OCR output for page {page_num} (custom_id={key})")
            continue

        pages = body.get("pages") or []
        if not pages or not isinstance(pages, list) or not isinstance(pages[0], dict):
            print(f"[mdx] invalid OCR output for page {page_num} (custom_id={key})")
            continue

        markdown = str(pages[0].get("markdown") or "").strip()
        mdx_content = _mdx_frontmatter(volume=vol_num, page_num=page_num) + markdown + "\n"

        page_file = output_mdx_dir / f"page_{page_num}.mdx"
        page_file.write_text(mdx_content, encoding="utf-8")

        if write_combined:
            combined_body = _strip_redundant_leading_headings(markdown, context)
            combined_parts.append(f"<!-- Page {page_num} -->\n\n{combined_body}".rstrip())

        context = _update_context_from_mdx(markdown, context)

    if write_combined and combined_parts:
        combined_path = output_mdx_dir / combined_filename
        combined_path.write_text(
            _combined_frontmatter(
                volume=vol_num,
                start_page=actual_start,
                end_page=actual_end,
                pdf_name=pdf_name,
            )
            + "\n\n".join(combined_parts).strip()
            + "\n",
            encoding="utf-8",
        )

    if not keep_batch_files:
        # Keep it best-effort and non-destructive (only remove the created jsonl).
        try:
            Path(batch_file).unlink(missing_ok=True)
        except Exception:
            pass

    print(f"Completed Volume {vol_num}. MDX saved in {output_mdx_dir}")


def _run_direct_pdf_to_mdx(
    *,
    pdf_path: str,
    vol_num: int,
    model: str,
    start_page: int,
    end_page: Optional[int],
    table_format: Optional[str],
    extract_header: bool,
    extract_footer: bool,
    include_image_base64: bool,
    write_combined: bool,
    combined_filename: str,
) -> None:
    client = _require_mistral_client()

    pdf_name = Path(pdf_path).stem
    output_mdx_dir = Path(f"vols/vol{vol_num}/mdx")
    output_mdx_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(pdf_path).read_bytes()
    doc_url = _data_url(mime_type="application/pdf", raw_bytes=raw)

    kwargs: dict[str, Any] = {}
    if table_format is not None:
        kwargs["table_format"] = table_format
    if extract_header:
        kwargs["extract_header"] = True
    if extract_footer:
        kwargs["extract_footer"] = True

    ocr_response = client.ocr.process(
        model=model,
        document={"type": "document_url", "document_url": doc_url},
        include_image_base64=include_image_base64,
        **kwargs,
    )

    pages = list(getattr(ocr_response, "pages", None) or [])
    if not pages:
        raise SystemExit("OCR response contains no pages.")

    total_pages = len(pages)
    start_page = max(1, start_page)
    end_page = total_pages if end_page is None else min(end_page, total_pages)
    if start_page > end_page:
        raise ValueError(f"Invalid page range: start_page={start_page} > end_page={end_page}")

    combined_parts: list[str] = []
    context = _OcrContext()

    for idx in range(start_page - 1, end_page):
        page_num = idx + 1
        page = pages[idx]
        markdown = str(getattr(page, "markdown", "") or "").strip()
        mdx_content = _mdx_frontmatter(volume=vol_num, page_num=page_num) + markdown + "\n"

        page_file = output_mdx_dir / f"page_{page_num}.mdx"
        page_file.write_text(mdx_content, encoding="utf-8")

        if write_combined:
            combined_body = _strip_redundant_leading_headings(markdown, context)
            combined_parts.append(f"<!-- Page {page_num} -->\n\n{combined_body}".rstrip())

        context = _update_context_from_mdx(markdown, context)

    if write_combined and combined_parts:
        combined_path = output_mdx_dir / combined_filename
        combined_path.write_text(
            _combined_frontmatter(
                volume=vol_num,
                start_page=start_page,
                end_page=end_page,
                pdf_name=pdf_name,
            )
            + "\n\n".join(combined_parts).strip()
            + "\n",
            encoding="utf-8",
        )

    print(f"Completed Volume {vol_num}. MDX saved in {output_mdx_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR PDF to MDX using Mistral OCR (mistral-ocr-latest)")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("vol_num", type=int, help="Volume number")

    parser.add_argument("--mode", choices=["batch", "direct"], default="batch")
    parser.add_argument("--model", default="mistral-ocr-latest")

    parser.add_argument("--start-page", type=int, default=1, help="1-based start page (default: 1)")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page (default: last)")

    parser.add_argument("--dpi", type=int, default=250, help="Rasterization DPI for --mode batch (default: 250)")
    parser.add_argument("--image-format", choices=["png", "jpeg"], default="png")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Batch polling interval (default: 5)")
    parser.add_argument("--keep-batch-files", action="store_true", help="Keep temp batch jsonl inputs")

    parser.add_argument("--table-format", choices=["markdown", "html"], default=None)
    parser.add_argument("--extract-header", action="store_true", default=False)
    parser.add_argument("--extract-footer", action="store_true", default=False)
    parser.add_argument("--no-image-base64", action="store_true", help="Disable include_image_base64")

    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not write vols/volN/mdx/combined.mdx (default: write it).",
    )
    parser.add_argument("--combined-filename", default="combined.mdx")

    args = parser.parse_args()

    include_image_base64 = not args.no_image_base64
    write_combined = not args.no_combined

    if args.mode == "batch":
        _run_batch_pdf_to_mdx(
            pdf_path=args.pdf_path,
            vol_num=args.vol_num,
            model=args.model,
            dpi=args.dpi,
            start_page=args.start_page,
            end_page=args.end_page,
            image_format=args.image_format,
            poll_seconds=args.poll_seconds,
            table_format=args.table_format,
            extract_header=args.extract_header,
            extract_footer=args.extract_footer,
            include_image_base64=include_image_base64,
            write_combined=write_combined,
            combined_filename=args.combined_filename,
            keep_batch_files=args.keep_batch_files,
        )
    else:
        _run_direct_pdf_to_mdx(
            pdf_path=args.pdf_path,
            vol_num=args.vol_num,
            model=args.model,
            start_page=args.start_page,
            end_page=args.end_page,
            table_format=args.table_format,
            extract_header=args.extract_header,
            extract_footer=args.extract_footer,
            include_image_base64=include_image_base64,
            write_combined=write_combined,
            combined_filename=args.combined_filename,
        )


if __name__ == "__main__":
    main()
