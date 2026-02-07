import pymupdf as fitz  # PyMuPDF
from pathlib import Path
import re
import argparse
from dataclasses import dataclass
from typing import Optional

class GLMOCRPipeline:
    """
    GLM-OCR only:
      - Rasterize PDF pages to images.
      - Use `zai-org/GLM-OCR` to produce MDX directly from the page image.
    """

    MODEL_ID = "zai-org/GLM-OCR"

    def __init__(self):
        self.model_id = self.MODEL_ID
        self._glm_loaded = False
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None

    @dataclass
    class OcrContext:
        major_heading: Optional[str] = None
        minor_heading: Optional[str] = None
        tail_text: str = ""

    def _strip_redundant_leading_headings(
        self, mdx: str, prev_context: "GLMOCRPipeline.OcrContext"
    ) -> str:
        """
        If a page starts by repeating the previous page's headings, drop those
        leading headings so the content naturally aggregates under a single
        heading when pages are concatenated.
        """
        lines = mdx.splitlines()
        idx = 0

        def skip_blank_lines(i: int) -> int:
            while i < len(lines) and not lines[i].strip():
                i += 1
            return i

        idx = skip_blank_lines(idx)

        # Allow stripping both a repeated major + minor heading.
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

    def _build_combined_frontmatter(
        self, *, volume: int, start_page: int, end_page: int, pdf_name: str
    ) -> str:
        return (
            "---\n"
            f"volume: {volume}\n"
            f"start_page: {start_page}\n"
            f"end_page: {end_page}\n"
            f"source_pdf: {pdf_name}\n"
            "---\n\n"
        )

    def _sanitize_ocr_output(self, text: str) -> str:
        """
        Best-effort cleanup for model artifacts (chat tokens, prompt leakage, placeholders).
        Keep conservative: only remove patterns that are extremely unlikely to be real page content.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Drop common chat special tokens (these often survive decoding depending on the model/tokenizer).
        text = re.sub(r"<\|[^|]+?\|>", "", text)

        # Drop generic wrapper prefixes.
        text = re.sub(
            r"^\s*(assistant|response|output)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Drop frequent hallucinated placeholders/instructions seen in bad generations.
        text = re.sub(
            r"(?im)^\s*-\s*If there is a title,.*?$",
            "",
            text,
        )
        text = re.sub(
            r"(?im)^\s*Final check of text\s*:?\s*$",
            "",
            text,
        )
        text = re.sub(
            r"(?im)^.*The image contains text from a PDF page\..*$",
            "",
            text,
        )

        # Drop green page markers often printed in the PDF margins/footers.
        text = re.sub(
            r"(?im)^\s*\(\s*Part\s*No\s*:\s*\d+\s*,\s*Page\s*No\s*:\s*\d+\s*\)\s*$",
            "",
            text,
        )

        # Normalize excessive blank lines after deletions.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _ensure_glm_loaded(self) -> None:
        if self._glm_loaded:
            return
        import torch  # lazy
        from transformers import AutoProcessor, AutoModelForImageTextToText  # lazy

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        if device.type == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        print(f"Using device: {device}")
        print(f"Loading processor and model: {self.model_id}")

        print("Initializing processor...")
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        print("Processor initialized.")

        print("Loading model weights (this may take a minute)...")
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        print(f"Model loaded on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")

        print("Setting model to eval mode...")
        model.eval()
        print("Model ready.")

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = model.device
        self._glm_loaded = True

    def _build_prompt(
        self,
        *,
        page_num: int,
        total_pages: int,
        context: "GLMOCRPipeline.OcrContext",
    ) -> str:
        # Keep this very short: GLM-OCR is sensitive to long prompts and may output the prompt itself.
        return "\n".join(
            [
                f"Page {page_num}/{total_pages}.",
                "Transcribe ALL visible text into Markdown/MDX.",
                "Output ONLY the transcription (no commentary, no 'References', no 'Final check').",
                "Do NOT add any text that is not on the page.",
                "Preserve paragraphs and line breaks.",
                "Ignore page markers like: (Part No : X, Page No: Y).",
                "If margin marker icons delimit quotes, wrap the passage in <Quran>...</Quran> (green) or <Hadith>...</Hadith> (blue); if unclear use <Quote>...</Quote>.",
                "If Q&A appears, wrap as <Question>...</Question> and <Answer>...</Answer>.",
            ]
        )

    def process_image(self, image_path, prompt: str):
        # GLM-OCR prompt is limited (see GLM-OCR.md). Use it for raw text recognition only.
        self._ensure_glm_loaded()
        torch = self._torch
        processor = self._processor
        model = self._model

        text_prompt = "Text Recognition:\n" + prompt.strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        print("Preparing inputs (applying chat template)...")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        print(f"Running model.generate for {image_path.name}...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        print("Generation complete. Decoding output...")

        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print("Decoding complete.")
        return self._sanitize_ocr_output(output_text)

    def _update_context_from_mdx(self, mdx: str, context: "GLMOCRPipeline.OcrContext") -> "GLMOCRPipeline.OcrContext":
        """
        Extracts headings and trailing text from MDX to update the OCR context.
        Resets minor heading if a new major heading is encountered.
        """
        major = context.major_heading
        minor = context.minor_heading

        for line in mdx.splitlines():
            if line.startswith("# "):
                new_major = line[2:].strip()
                if new_major:
                    if new_major != major:
                        major = new_major
                        minor = None  # Reset minor heading for a new major section
            elif line.startswith("## "):
                new_minor = line[3:].strip()
                if new_minor:
                    minor = new_minor

        # Keep last non-empty tail for continuity.
        tail = re.sub(r"\s+", " ", mdx).strip()
        tail = tail[-800:] if tail else ""
        return self.OcrContext(
            major_heading=major,
            minor_heading=minor,
            tail_text=tail,
        )

    def convert_to_mdx(self, ocr_text, volume, page_num):
        # Post-processing to ensure clean MDX
        # 1. Add frontmatter
        frontmatter = f"""---
volume: {volume}
page: {page_num}
---

"""
        
        # 2. Heuristic boundary detection for Quran if the model missed it
        # (e.g., looking for typical brackets or Arabic scripts)
        processed_text = ocr_text

        return frontmatter + processed_text

    def process_pdf(
        self,
        pdf_path,
        vol_num,
        *,
        dpi=300,
        start_page=1,
        end_page=None,
        write_combined=True,
        combined_filename="combined.mdx",
    ):
        pdf_name = Path(pdf_path).stem
        temp_img_dir = Path(f"temp_images_{pdf_name}")
        output_mdx_dir = Path(f"vols/vol{vol_num}/mdx")
        output_mdx_dir.mkdir(parents=True, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        end_page = total_pages if end_page is None else min(end_page, total_pages)
        start_page = max(1, start_page)
        if start_page > end_page:
            raise ValueError(f"Invalid page range: start_page={start_page} > end_page={end_page}")

        temp_img_dir.mkdir(parents=True, exist_ok=True)

        combined_parts = []
        context = self.OcrContext()
        
        for i in range(start_page - 1, end_page):
            page_num = i + 1
            page = doc.load_page(i)
            print(f"Rendering page {page_num}/{total_pages} for GLM-OCR...")
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            img_path = temp_img_dir / f"page_{page_num}.png"
            pix.save(str(img_path))

            prompt = self._build_prompt(
                page_num=page_num,
                total_pages=total_pages,
                context=context,
            )
            print(f"GLM-OCR page {page_num}/{total_pages}...")
            mdx_body = self.process_image(img_path, prompt)

            mdx_content = self.convert_to_mdx(mdx_body, vol_num, page_num)
            
            page_file = output_mdx_dir / f"page_{page_num}.mdx"
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(mdx_content)
            
            if write_combined:
                combined_body = self._strip_redundant_leading_headings(mdx_body, context)
                combined_parts.append(f"<!-- Page {page_num} -->\n\n{combined_body}".rstrip())
            context = self._update_context_from_mdx(mdx_body, context)
            
        if write_combined and combined_parts:
            combined_path = output_mdx_dir / combined_filename
            with open(combined_path, "w", encoding="utf-8") as f:
                f.write(
                    self._build_combined_frontmatter(
                        volume=vol_num,
                        start_page=start_page,
                        end_page=end_page,
                        pdf_name=pdf_name,
                    )
                )
                f.write("\n\n".join(combined_parts).strip() + "\n")
        
        print(f"Completed processing Volume {vol_num}. MDX files saved in {output_mdx_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR PDF to MDX using GLM-OCR")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("vol_num", type=int, help="Volume number")
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI (default: 300)")
    parser.add_argument("--start-page", type=int, default=1, help="1-based start page (default: 1)")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page (default: last)")
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not write vols/volN/mdx/combined.mdx (default: write it).",
    )
    args = parser.parse_args()
    
    pipeline = GLMOCRPipeline()
    pipeline.process_pdf(
        args.pdf_path,
        args.vol_num,
        dpi=args.dpi,
        start_page=args.start_page,
        end_page=args.end_page,
        write_combined=not args.no_combined,
    )
