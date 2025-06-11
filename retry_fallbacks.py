#!/usr/bin/env python3
"""
Script to identify and retry fallbacked structures from parser.py output.
Fallback content has headings like "Page <n> content" indicating LLM processing failed.
"""

import json
import asyncio
import aiohttp
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FallbackSection:
    """Represents a fallbacked section that needs retry"""
    page_number: int
    section_index: int
    heading_text: str
    content_items: List[Dict[str, Any]]
    file_source: str
    confidence_scores: List[float]
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence of content items"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    @property
    def needs_retry(self) -> bool:
        """Determine if this section needs retry based on confidence and patterns"""
        # Check for fallback patterns
        if "Page" in self.heading_text and "Content" in self.heading_text:
            return True
        
        # Check for low confidence
        if self.avg_confidence < 0.6:
            return True
            
        # Check for fallback classification notes
        fallback_notes = [item.get('notes', '') for item in self.content_items]
        if any('fallback' in note or 'memory_prediction' in note for note in fallback_notes):
            return True
            
        return False

class FallbackRetryProcessor:
    """Process and retry fallbacked structures from parser.py output"""
    
    def __init__(self, output_dir: str, api_key: str = None, pdf_path: str = None):
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.pdf_path = pdf_path
        self.fallback_sections: List[FallbackSection] = []
        self.retry_results: Dict[int, Dict] = {}
        
        # Initialize PDF document if path provided
        self.doc = None
        if pdf_path and Path(pdf_path).exists():
            try:
                import fitz
                self.doc = fitz.open(pdf_path)
                logger.info(f"Loaded PDF: {pdf_path} ({len(self.doc)} pages)")
            except Exception as e:
                logger.warning(f"Could not load PDF {pdf_path}: {e}")
        
        # Load memory from parser.py output
        self.memory = self._load_parsing_memory()
        
        # LLM settings
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # Patterns to identify fallback content
        self.fallback_patterns = [
            r"Page \d+ [Cc]ontent",
            r"Page \d+ [Ff]allback",
            r"Emergency [Ff]allback",
            r"Fallback [Ss]tructure"
        ]
        
        logger.info(f"Initialized fallback retry processor for: {self.output_dir}")
    
    def scan_for_fallbacks(self, specific_file: str = None) -> List[FallbackSection]:
        """Scan all output files for fallbacked structures"""
        logger.info("Scanning for fallbacked structures...")
        
        fallbacks = []
        
        if specific_file:
            # Scan specific file provided by user
            file_path = Path(specific_file)
            if file_path.exists():
                logger.info(f"Scanning specific file: {file_path}...")
                fallbacks.extend(self._scan_file_for_fallbacks(file_path, file_path.name))
            else:
                logger.error(f"Specified file does not exist: {specific_file}")
                return []
        else:
            # Scan main output files in the directory
            main_files = [
                "llm_parsed_complete.json",
                "react_islamic_components.json"
            ]
            
            for filename in main_files:
                file_path = self.output_dir / filename
                if file_path.exists():
                    logger.info(f"Scanning {filename}...")
                    fallbacks.extend(self._scan_file_for_fallbacks(file_path, filename))
            
            # Scan intermediate batch files
            batch_files = list(self.output_dir.glob("intermediate_batch_*.json"))
            for batch_file in sorted(batch_files):
                logger.info(f"Scanning {batch_file.name}...")
                fallbacks.extend(self._scan_file_for_fallbacks(batch_file, batch_file.name))
        
        self.fallback_sections = fallbacks
        logger.info(f"Found {len(fallbacks)} fallbacked sections")
        
        return fallbacks
    
    def _scan_file_for_fallbacks(self, file_path: Path, source_name: str) -> List[FallbackSection]:
        """Scan a single file for fallback structures"""
        fallbacks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read file in chunks to avoid memory issues with large files
                content = f.read()
                
            # Parse JSON
            data = json.loads(content)
            
            # Extract sections based on file structure
            sections = []
            if 'sections' in data:
                sections = data['sections']
            elif 'sections' in data and isinstance(data['sections'], list):
                sections = data['sections']
            
            # Scan each section
            for idx, section in enumerate(sections):
                if self._is_fallback_section(section):
                    fallback = self._create_fallback_section(section, idx, source_name)
                    if fallback:
                        fallbacks.append(fallback)
                        
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON from {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
        
        return fallbacks
    
    def _is_fallback_section(self, section: Dict) -> bool:
        """Check if a section is a fallback structure"""
        # Check heading for fallback patterns
        heading = section.get('heading', {})
        heading_text = heading.get('text', '')
        
        for pattern in self.fallback_patterns:
            if re.search(pattern, heading_text, re.IGNORECASE):
                return True
        
        # Check content items for fallback indicators
        content_items = section.get('content_items', [])
        for item in content_items:
            notes = item.get('notes', '')
            if 'fallback' in notes.lower() or 'memory_prediction' in notes.lower():
                return True
            
            # Check for low confidence
            confidence = item.get('confidence', 1.0)
            if confidence < 0.6:
                return True
        
        # Check processing method in metadata
        if 'metadata' in section:
            processing_method = section['metadata'].get('processing_method', '')
            if 'fallback' in processing_method.lower():
                return True
        
        return False
    
    def _create_fallback_section(self, section: Dict, index: int, source: str) -> FallbackSection:
        """Create a FallbackSection object from section data"""
        try:
            heading = section.get('heading', {})
            heading_text = heading.get('text', '')
            
            # Extract page number from heading
            page_match = re.search(r'Page (\d+)', heading_text)
            page_number = int(page_match.group(1)) if page_match else heading.get('page_number', 0)
            
            content_items = section.get('content_items', [])
            confidence_scores = [item.get('confidence', 0.5) for item in content_items]
            
            return FallbackSection(
                page_number=page_number,
                section_index=index,
                heading_text=heading_text,
                content_items=content_items,
                file_source=source,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback section: {e}")
            return None
    
    def analyze_fallbacks(self) -> Dict[str, Any]:
        """Analyze the fallbacked sections to understand patterns"""
        if not self.fallback_sections:
            return {"error": "No fallback sections found"}
        
        analysis = {
            "total_fallbacks": len(self.fallback_sections),
            "pages_affected": sorted(list(set(fb.page_number for fb in self.fallback_sections))),
            "confidence_distribution": {},
            "fallback_types": {},
            "source_files": {},
            "retry_priority": []
        }
        
        # Analyze confidence distribution
        all_confidences = []
        for fb in self.fallback_sections:
            all_confidences.extend(fb.confidence_scores)
        
        if all_confidences:
            analysis["confidence_distribution"] = {
                "min": min(all_confidences),
                "max": max(all_confidences),
                "avg": sum(all_confidences) / len(all_confidences),
                "below_0.5": len([c for c in all_confidences if c < 0.5]),
                "below_0.6": len([c for c in all_confidences if c < 0.6])
            }
        
        # Analyze fallback types
        for fb in self.fallback_sections:
            # Categorize by heading pattern
            if "Page" in fb.heading_text and "Content" in fb.heading_text:
                fb_type = "page_content_fallback"
            elif "Emergency" in fb.heading_text:
                fb_type = "emergency_fallback"
            else:
                fb_type = "low_confidence_fallback"
            
            analysis["fallback_types"][fb_type] = analysis["fallback_types"].get(fb_type, 0) + 1
        
        # Analyze source files
        for fb in self.fallback_sections:
            source = fb.file_source
            analysis["source_files"][source] = analysis["source_files"].get(source, 0) + 1
        
        # Create retry priority list
        priority_fallbacks = sorted(self.fallback_sections, 
                                  key=lambda x: (x.avg_confidence, -x.page_number))
        
        analysis["retry_priority"] = [
            {
                "page_number": fb.page_number,
                "section_index": fb.section_index,
                "avg_confidence": fb.avg_confidence,
                "content_items_count": len(fb.content_items),
                "source": fb.file_source,
                "needs_retry": fb.needs_retry
            }
            for fb in priority_fallbacks[:20]  # Top 20 priority items
        ]
        
        return analysis
    
    def extract_retry_batches(self, max_pages_per_batch: int = 5) -> List[Dict[str, Any]]:
        """Extract page content for retry processing in batches"""
        if not self.fallback_sections:
            logger.warning("No fallback sections to extract")
            return []
        
        # Group fallbacks by page number
        pages_to_retry = {}
        for fb in self.fallback_sections:
            if fb.needs_retry:
                if fb.page_number not in pages_to_retry:
                    pages_to_retry[fb.page_number] = []
                pages_to_retry[fb.page_number].append(fb)
        
        # Create batches
        batches = []
        page_numbers = sorted(pages_to_retry.keys())
        
        for i in range(0, len(page_numbers), max_pages_per_batch):
            batch_pages = page_numbers[i:i + max_pages_per_batch]
            
            batch = {
                "batch_id": len(batches) + 1,
                "pages": batch_pages,
                "fallback_sections": [],
                "total_content_items": 0,
                "avg_confidence": 0.0,
                "retry_reason": "fallback_detected"
            }
            
            all_confidences = []
            for page_num in batch_pages:
                for fb in pages_to_retry[page_num]:
                    batch["fallback_sections"].append({
                        "page_number": fb.page_number,
                        "section_index": fb.section_index,
                        "heading": fb.heading_text,
                        "content_items": fb.content_items,
                        "source": fb.file_source
                    })
                    batch["total_content_items"] += len(fb.content_items)
                    all_confidences.extend(fb.confidence_scores)
            
            if all_confidences:
                batch["avg_confidence"] = sum(all_confidences) / len(all_confidences)
            
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} retry batches for {len(page_numbers)} pages")
        return batches
    
    def _load_parsing_memory(self) -> Dict:
        """Load existing parsing memory from parser.py output"""
        memory_file = self.output_dir / "parsing_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load parsing memory: {e}")
        return {}
    
    def extract_page_content_from_pdf(self, page_num: int) -> Dict[str, Any]:
        """Extract raw text content from a PDF page (similar to parser.py)"""
        if not self.doc:
            logger.error("PDF document not loaded")
            return {"page_number": page_num, "content_blocks": []}
        
        try:
            page = self.doc[page_num - 1]  # Convert to 0-based index
            page_dict = page.get_text("dict")
            page_width = page.rect.width
            
            content_blocks = []
            
            for block in page_dict["blocks"]:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    line_bbox = None
                    max_font_size = 0
                    primary_font = ""
                    
                    for span in line["spans"]:
                        line_text += span["text"]
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                            primary_font = span["font"]
                        
                        if line_bbox is None:
                            line_bbox = list(span["bbox"])
                        else:
                            span_bbox = span["bbox"]
                            line_bbox[0] = min(line_bbox[0], span_bbox[0])
                            line_bbox[1] = min(line_bbox[1], span_bbox[1])
                            line_bbox[2] = max(line_bbox[2], span_bbox[2])
                            line_bbox[3] = max(line_bbox[3], span_bbox[3])
                    
                    line_text = line_text.strip()
                    if len(line_text) < 2:
                        continue
                    
                    # Calculate positioning hints
                    text_center = (line_bbox[0] + line_bbox[2]) / 2
                    page_center = page_width / 2
                    is_centered = abs(text_center - page_center) <= 80
                    
                    content_blocks.append({
                        "text": line_text,
                        "font": primary_font,
                        "size": max_font_size,
                        "bbox": line_bbox,
                        "is_centered": is_centered,
                        "is_bold": "Bold" in primary_font,
                        "y_position": line_bbox[1]
                    })
            
            # Sort by Y position (top to bottom)
            content_blocks.sort(key=lambda x: x["y_position"])
            
            return {
                "page_number": page_num,
                "content_blocks": content_blocks,
                "page_width": page_width,
                "page_height": page.rect.height
            }
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {e}")
            return {"page_number": page_num, "content_blocks": []}
    
    def create_retry_prompt(self, page_contents: List[Dict]) -> str:
        """Create a structured prompt for LLM retry analysis"""
        
        # Content types from parser.py
        content_types = {
            "heading_major": "Major section headings (Foreword, Introduction, Chapter titles)",
            "heading_minor": "Subsection headings and topic titles",
            "heading_question": "Question numbers or fatwa identifiers",
            "quran_verse": "Verses from the Quran with references",
            "hadith": "Prophetic traditions with chain of narration",
            "fatwa_ruling": "Religious ruling or legal opinion",
            "question": "Questions posed to the scholar",
            "answer": "Scholar's response to questions",
            "arabic_text": "Arabic text (often with transliteration)",
            "transliteration": "Romanized Arabic text",
            "translation": "English translation of Arabic",
            "signature": "Author signatures and attributions",
            "citation": "References to books, scholars, or sources",
            "footnote": "Explanatory notes and clarifications",
            "paragraph": "Regular explanatory text",
            "list_item": "Enumerated points or conditions",
            "navigation": "Part/page/volume indicators"
        }
        
        # Combine text from multiple pages for context
        combined_text = ""
        
        for page_data in page_contents:
            page_num = page_data["page_number"]
            combined_text += f"\n=== PAGE {page_num} ===\n"
            
            for block in page_data["content_blocks"]:
                # Add formatting hints
                formatting_hints = []
                if block["is_bold"]:
                    formatting_hints.append("BOLD")
                if block["is_centered"]:
                    formatting_hints.append("CENTERED")
                if block["size"] > 15:
                    formatting_hints.append("LARGE_FONT")
                
                hint_str = f" [{', '.join(formatting_hints)}]" if formatting_hints else ""
                combined_text += f"{block['text']}{hint_str}\n"
        
        content_types_desc = "\n".join([f"- {k}: {v}" for k, v in content_types.items()])
        
        # Build context from memory if available
        context_info = ""
        if self.memory:
            context_info = "MEMORY CONTEXT: Using learned patterns from previous processing\n"
        
        prompt = f"""
You are analyzing Islamic scholarly text from "Majmoo'al-Fatawa of Ibn Bazz". This is a RETRY of previously failed content classification. Your task is to identify and classify different types of content according to their semantic meaning and structure.

{context_info}

CONTENT TYPES TO IDENTIFY:
{content_types_desc}

TEXT TO ANALYZE (RETRY):
{combined_text}

INSTRUCTIONS:
1. This is a RETRY of content that previously failed LLM processing
2. Pay special attention to proper classification of each content piece
3. Focus on:
   - CENTERED and BOLD text are often headings
   - "Foreword", "Introduction", "Chapter" are major headings
   - Questions usually start with numbers or "Question:"
   - Arabic text vs transliteration vs translation
   - Signature lines and attributions
   - Quranic verses and Hadith citations

4. Group related content under appropriate headings
5. Preserve the logical flow and hierarchy
6. Include confidence scores for your classifications
7. Provide higher confidence scores than the original failed attempt

RESPOND ONLY WITH VALID JSON in this exact format:
{{
  "sections": [
    {{
      "type": "content_section",
      "heading": {{
        "text": "exact heading text",
        "content_type": "heading_major|heading_minor|heading_question",
        "page_number": page_number,
        "confidence": 0.0-1.0
      }},
      "content_items": [
        {{
          "text": "exact text content",
          "content_type": "paragraph|quran_verse|hadith|fatwa_ruling|question|answer|etc",
          "page_number": page_number,
          "confidence": 0.0-1.0,
          "notes": "retry_processing",
          "retry_attempt": true
        }}
      ]
    }}
  ],
  "metadata": {{
    "pages_analyzed": [list of page numbers],
    "total_sections_found": number,
    "primary_language": "english|arabic|mixed",
    "processing_method": "llm_retry",
    "retry_timestamp": "{datetime.now().isoformat()}"
  }}
}}

CRITICAL: Return ONLY the JSON response, no additional text or explanation.
"""
        return prompt
    
    async def call_anthropic_api(self, session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Make API call to Anthropic Claude with retry logic"""
        if not self.api_key:
            logger.error("No API key provided for LLM retry")
            return {"error": "No API key provided"}
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 30000,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM retry API call attempt {attempt + 1}/{max_retries}")
                
                async with session.post(self.base_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["content"][0]["text"]
                        
                        # Try to parse JSON from response
                        try:
                            parsed_result = json.loads(content)
                            logger.info(f"LLM retry API call successful on attempt {attempt + 1}")
                            return parsed_result
                            
                        except json.JSONDecodeError as json_error:
                            logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {json_error}")
                            # Extract JSON if wrapped in other text
                            start_idx = content.find('{')
                            end_idx = content.rfind('}') + 1
                            if start_idx != -1 and end_idx != 0:
                                try:
                                    parsed_result = json.loads(content[start_idx:end_idx])
                                    logger.info(f"JSON extraction successful on attempt {attempt + 1}")
                                    return parsed_result
                                except json.JSONDecodeError:
                                    pass
                            
                            # If this is the last attempt, return error
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to parse JSON from response: {content[:500]}...")
                                return {"error": "JSON parsing failed", "raw_response": content[:1000]}
                    
                    elif response.status == 429:  # Rate limit
                        error_text = await response.text()
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited (429) on attempt {attempt + 1}, waiting {wait_time}s: {error_text}")
                        last_error = f"Rate limit: {error_text}"
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(wait_time)
                            continue
                    
                    else:  # Other errors
                        error_text = await response.text()
                        logger.error(f"API error ({response.status}) on attempt {attempt + 1}: {error_text}")
                        return {"error": f"API error: {response.status}", "details": error_text}
                        
            except Exception as e:
                wait_time = 2 ** attempt
                logger.error(f"Error on attempt {attempt + 1}, waiting {wait_time}s: {str(e)}")
                last_error = f"Error: {str(e)}"
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    continue
        
        # All retries failed
        logger.error(f"All {max_retries} LLM retry attempts failed. Last error: {last_error}")
        return {"error": "All retries failed", "details": last_error or "Unknown error"}
    
    async def retry_fallback_pages(self, page_numbers: List[int]) -> Dict[str, Any]:
        """Retry processing specific pages with LLM"""
        if not self.doc:
            logger.error("PDF document not loaded - cannot retry pages")
            return {"error": "PDF not loaded"}
        
        if not self.api_key:
            logger.error("API key not provided - cannot retry with LLM")
            return {"error": "No API key"}
        
        logger.info(f"Retrying LLM processing for pages: {page_numbers}")
        
        # Extract page content from PDF
        page_contents = []
        for page_num in page_numbers:
            page_content = self.extract_page_content_from_pdf(page_num)
            if page_content["content_blocks"]:
                page_contents.append(page_content)
        
        if not page_contents:
            logger.warning(f"No content extracted from pages {page_numbers}")
            return {"error": "No content extracted"}
        
        # Create retry prompt
        prompt = self.create_retry_prompt(page_contents)
        
        # Call LLM API
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=180, connect=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            result = await self.call_anthropic_api(session, prompt)
            
            if "error" in result:
                logger.error(f"LLM retry failed for pages {page_numbers}: {result}")
                return result
            
            logger.info(f"Successfully retried pages {page_numbers} with LLM")
            
            # Add retry metadata
            result["retry_info"] = {
                "original_pages": page_numbers,
                "retry_timestamp": datetime.now().isoformat(),
                "retry_method": "llm_api_call"
            }
            
            return result
    
    async def process_retry_batches(self, batches: List[Dict]) -> Dict[str, Any]:
        """Process all retry batches with LLM"""
        if not self.api_key:
            logger.error("API key required for LLM retry processing")
            return {"error": "No API key provided"}
        
        logger.info(f"Starting LLM retry processing for {len(batches)} batches")
        
        retry_results = []
        successful_retries = 0
        failed_retries = 0
        
        for batch in batches:
            batch_id = batch["batch_id"]
            pages = batch["pages"]
            
            logger.info(f"Processing retry batch {batch_id}: pages {pages}")
            
            try:
                result = await self.retry_fallback_pages(pages)
                
                if "error" not in result:
                    successful_retries += 1
                    retry_results.append({
                        "batch_id": batch_id,
                        "pages": pages,
                        "status": "success",
                        "result": result
                    })
                    logger.info(f"Batch {batch_id} retry successful")
                else:
                    failed_retries += 1
                    retry_results.append({
                        "batch_id": batch_id,
                        "pages": pages,
                        "status": "failed",
                        "error": result["error"]
                    })
                    logger.warning(f"Batch {batch_id} retry failed: {result['error']}")
                
                # Rate limiting between batches
                await asyncio.sleep(2)
                
            except Exception as e:
                failed_retries += 1
                logger.error(f"Critical error in batch {batch_id}: {e}")
                retry_results.append({
                    "batch_id": batch_id,
                    "pages": pages,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save retry results
        retry_summary = {
            "retry_summary": {
                "total_batches": len(batches),
                "successful_retries": successful_retries,
                "failed_retries": failed_retries,
                "success_rate": f"{(successful_retries / len(batches) * 100):.1f}%" if batches else "0%",
                "processing_timestamp": datetime.now().isoformat()
            },
            "batch_results": retry_results
        }
        
        # Save results
        retry_file = self.output_dir / "retry_results.json"
        with open(retry_file, 'w', encoding='utf-8') as f:
            json.dump(retry_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Retry processing complete: {successful_retries}/{len(batches)} successful")
        logger.info(f"Results saved to: {retry_file}")
        
        # Apply successful fixes to original files
        if successful_retries > 0:
            logger.info("Applying successful retry results to original files...")
            fix_summary = self.apply_retry_fixes(retry_results)
            retry_summary["fix_summary"] = fix_summary
        
        return retry_summary
    
    def apply_retry_fixes(self, retry_results: List[Dict]) -> Dict[str, Any]:
        """Apply successful retry results to original files, replacing fallbacked sections"""
        logger.info("Starting to apply retry fixes to original files...")
        
        fix_summary = {
            "files_updated": [],
            "sections_fixed": 0,
            "pages_fixed": [],
            "errors": [],
            "backup_created": False
        }
        
        # Group successful results by source file
        fixes_by_file = {}
        
        for retry_result in retry_results:
            if retry_result["status"] != "success":
                continue
                
            result_data = retry_result["result"]
            pages = retry_result["pages"]
            
            # Find which fallback sections correspond to these pages
            for fb in self.fallback_sections:
                if fb.page_number in pages and fb.needs_retry:
                    source_file = fb.file_source
                    
                    if source_file not in fixes_by_file:
                        fixes_by_file[source_file] = []
                    
                    fixes_by_file[source_file].append({
                        "fallback_section": fb,
                        "retry_result": result_data,
                        "pages": pages
                    })
        
        # Apply fixes to each file
        for source_file, fixes in fixes_by_file.items():
            try:
                file_path = self.output_dir / source_file
                if not file_path.exists():
                    logger.error(f"Source file not found: {file_path}")
                    fix_summary["errors"].append(f"File not found: {source_file}")
                    continue
                
                # Create backup
                backup_path = self.output_dir / f"{source_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Load original file
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # Create backup
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, indent=2, ensure_ascii=False)
                fix_summary["backup_created"] = True
                
                logger.info(f"Created backup: {backup_path}")
                
                # Apply fixes to the data
                updated_data = self._apply_fixes_to_data(original_data, fixes)
                
                # Save updated file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                
                fix_summary["files_updated"].append(source_file)
                fix_summary["sections_fixed"] += len(fixes)
                
                # Track pages fixed
                for fix in fixes:
                    fix_summary["pages_fixed"].extend(fix["pages"])
                
                logger.info(f"Successfully updated {source_file} with {len(fixes)} fixes")
                
            except Exception as e:
                error_msg = f"Error updating {source_file}: {str(e)}"
                logger.error(error_msg)
                fix_summary["errors"].append(error_msg)
        
        # Remove duplicates from pages_fixed
        fix_summary["pages_fixed"] = sorted(list(set(fix_summary["pages_fixed"])))
        
        # Save fix summary
        fix_summary_file = self.output_dir / "fix_summary.json"
        with open(fix_summary_file, 'w', encoding='utf-8') as f:
            json.dump(fix_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fix summary saved to: {fix_summary_file}")
        logger.info(f"Applied fixes: {fix_summary['sections_fixed']} sections in {len(fix_summary['files_updated'])} files")
        
        return fix_summary
    
    def _apply_fixes_to_data(self, original_data: Dict, fixes: List[Dict]) -> Dict:
        """Apply retry fixes to the original data structure"""
        updated_data = original_data.copy()
        
        if 'sections' not in updated_data:
            logger.warning("No 'sections' key found in original data")
            return updated_data
        
        sections = updated_data['sections']
        
        # Create a mapping of page numbers to retry results
        page_to_retry_sections = {}
        for fix in fixes:
            retry_result = fix["retry_result"]
            pages = fix["pages"]
            
            if "sections" in retry_result:
                for page_num in pages:
                    if page_num not in page_to_retry_sections:
                        page_to_retry_sections[page_num] = []
                    
                    # Find sections for this page in the retry result
                    for retry_section in retry_result["sections"]:
                        if retry_section.get("heading", {}).get("page_number") == page_num:
                            page_to_retry_sections[page_num].append(retry_section)
        
        # Replace fallbacked sections with retry results
        sections_to_remove = []
        sections_to_add = []
        
        for i, section in enumerate(sections):
            # Check if this section is a fallback that we have a fix for
            heading = section.get('heading', {})
            heading_text = heading.get('text', '')
            
            # Extract page number from section
            page_match = re.search(r'Page (\d+)', heading_text)
            if page_match:
                page_number = int(page_match.group(1))
                
                # Check if this is a fallback section we want to replace
                if self._is_fallback_section(section) and page_number in page_to_retry_sections:
                    logger.info(f"Replacing fallback section for page {page_number} at index {i}")
                    
                    # Mark for removal
                    sections_to_remove.append(i)
                    
                    # Add retry sections for this page
                    retry_sections = page_to_retry_sections[page_number]
                    for retry_section in retry_sections:
                        # Add metadata to indicate this was fixed
                        if "metadata" not in retry_section:
                            retry_section["metadata"] = {}
                        
                        retry_section["metadata"].update({
                            "processing_method": "llm_retry_fixed",
                            "original_section_index": i,
                            "fix_timestamp": datetime.now().isoformat(),
                            "replaced_fallback": True
                        })
                        
                        sections_to_add.append((i, retry_section))
        
        # Remove fallback sections (in reverse order to maintain indices)
        for index in sorted(sections_to_remove, reverse=True):
            del sections[index]
        
        # Insert new sections at appropriate positions
        # Sort by original position to maintain order
        sections_to_add.sort(key=lambda x: x[0])
        
        for original_index, new_section in sections_to_add:
            # Find the best insertion point (may have shifted due to deletions)
            insert_index = min(original_index, len(sections))
            sections.insert(insert_index, new_section)
        
        # Update metadata
        if "metadata" not in updated_data:
            updated_data["metadata"] = {}
        
        updated_data["metadata"].update({
            "last_fix_applied": datetime.now().isoformat(),
            "fallback_fixes_applied": len(sections_to_remove),
            "retry_sections_added": len(sections_to_add)
        })
        
        logger.info(f"Replaced {len(sections_to_remove)} fallback sections with {len(sections_to_add)} retry sections")
        
        return updated_data
    
    def generate_retry_script(self, batches: List[Dict], output_file: str = "retry_commands.sh"):
        """Generate a script to retry the fallbacked sections"""
        script_path = self.output_dir / output_file
        
        script_content = f"""#!/bin/bash
# Auto-generated retry script for fallbacked sections
# Generated on: {datetime.now().isoformat()}
# Total batches to retry: {len(batches)}

set -e  # Exit on error

echo "Starting fallback retry processing..."
echo "Total batches: {len(batches)}"

"""
        
        for batch in batches:
            pages_str = ",".join(map(str, batch["pages"]))
            script_content += f"""
echo "Processing batch {batch['batch_id']}: pages {pages_str}"
echo "  - Pages: {len(batch['pages'])}"
echo "  - Content items: {batch['total_content_items']}"
echo "  - Avg confidence: {batch['avg_confidence']:.3f}"

# You can customize this command based on your parser.py interface
# python parser.py --retry-pages {pages_str} --output-suffix "_retry_{batch['batch_id']}"

"""
        
        script_content += """
echo "Fallback retry processing complete!"
echo "Check the output directory for retry results."
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        logger.info(f"Generated retry script: {script_path}")
        return script_path
    
    def save_retry_analysis(self, analysis: Dict, batches: List[Dict]):
        """Save the retry analysis and batch information"""
        
        # Save analysis
        analysis_file = self.output_dir / "fallback_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Save retry batches
        batches_file = self.output_dir / "retry_batches.json"
        with open(batches_file, 'w', encoding='utf-8') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "total_batches": len(batches),
                "batches": batches
            }, f, indent=2, ensure_ascii=False)
        
        # Save detailed fallback sections
        fallbacks_file = self.output_dir / "fallback_sections_detailed.json"
        fallback_data = []
        for fb in self.fallback_sections:
            fallback_data.append({
                "page_number": fb.page_number,
                "section_index": fb.section_index,
                "heading_text": fb.heading_text,
                "content_items_count": len(fb.content_items),
                "avg_confidence": fb.avg_confidence,
                "needs_retry": fb.needs_retry,
                "file_source": fb.file_source,
                "content_preview": fb.content_items[0].get('text', '')[:100] + "..." if fb.content_items else ""
            })
        
        with open(fallbacks_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_fallbacks": len(fallback_data),
                "fallbacks": fallback_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved analysis to: {analysis_file}")
        logger.info(f"Saved retry batches to: {batches_file}")
        logger.info(f"Saved detailed fallbacks to: {fallbacks_file}")
    
    def print_summary(self, analysis: Dict, batches: List[Dict]):
        """Print a summary of the fallback analysis"""
        print("\n" + "="*60)
        print("FALLBACK RETRY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  • Total fallback sections found: {analysis['total_fallbacks']}")
        print(f"  • Pages affected: {len(analysis['pages_affected'])}")
        print(f"  • Retry batches created: {len(batches)}")
        
        if analysis.get('confidence_distribution'):
            conf = analysis['confidence_distribution']
            print(f"\n📈 CONFIDENCE ANALYSIS:")
            print(f"  • Average confidence: {conf['avg']:.3f}")
            print(f"  • Range: {conf['min']:.3f} - {conf['max']:.3f}")
            print(f"  • Items below 0.5 confidence: {conf['below_0.5']}")
            print(f"  • Items below 0.6 confidence: {conf['below_0.6']}")
        
        print(f"\n🔍 FALLBACK TYPES:")
        for fb_type, count in analysis['fallback_types'].items():
            print(f"  • {fb_type}: {count}")
        
        print(f"\n📁 SOURCE FILES:")
        for source, count in analysis['source_files'].items():
            print(f"  • {source}: {count} sections")
        
        print(f"\n🎯 TOP PRIORITY PAGES FOR RETRY:")
        for item in analysis['retry_priority'][:10]:
            status = "✅ NEEDS RETRY" if item['needs_retry'] else "⚠️  LOW PRIORITY"
            print(f"  • Page {item['page_number']}: conf={item['avg_confidence']:.3f}, items={item['content_items_count']} {status}")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"  1. Review generated files:")
        print(f"     - fallback_analysis.json")
        print(f"     - retry_batches.json") 
        print(f"     - fallback_sections_detailed.json")
        print(f"     - retry_commands.sh")
        print(f"  2. Execute retry script or manually process priority pages")
        print(f"  3. Update parser.py with improved patterns if needed")
        
        print("\n" + "="*60)
    
    def print_fix_summary(self, fix_summary: Dict):
        """Print a summary of the fixes applied"""
        print("\n" + "="*60)
        print("FALLBACK FIXES APPLIED SUMMARY")
        print("="*60)
        
        print(f"\n🔧 FIX RESULTS:")
        print(f"  • Files updated: {len(fix_summary['files_updated'])}")
        print(f"  • Sections fixed: {fix_summary['sections_fixed']}")
        print(f"  • Pages fixed: {len(fix_summary['pages_fixed'])}")
        print(f"  • Backup created: {'✅' if fix_summary['backup_created'] else '❌'}")
        
        if fix_summary['files_updated']:
            print(f"\n📁 UPDATED FILES:")
            for file_name in fix_summary['files_updated']:
                print(f"  • {file_name}")
        
        if fix_summary['pages_fixed']:
            print(f"\n📄 PAGES FIXED:")
            pages_str = ", ".join(map(str, sorted(fix_summary['pages_fixed'])))
            print(f"  • {pages_str}")
        
        if fix_summary['errors']:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in fix_summary['errors']:
                print(f"  • {error}")
        else:
            print(f"\n✅ No errors encountered during fix application")
        
        print(f"\n📋 WHAT WAS DONE:")
        print(f"  1. Created backup files with timestamp")
        print(f"  2. Replaced fallbacked sections with LLM retry results")
        print(f"  3. Updated original files with corrected content")
        print(f"  4. Added metadata to track fixes applied")
        print(f"  5. Saved fix summary to fix_summary.json")
        
        print("\n" + "="*60)

async def main():
    """Main function to run fallback analysis and retry"""
    parser = argparse.ArgumentParser(description="Analyze and retry fallbacked structures from parser.py output")
    parser.add_argument("input_path", help="Path to JSON file or directory containing parser.py output files")
    parser.add_argument("--api-key", help="Anthropic API key for retry processing")
    parser.add_argument("--pdf-path", help="Path to original PDF file for retry processing")
    parser.add_argument("--batch-size", type=int, default=1, help="Pages per retry batch")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence threshold")
    parser.add_argument("--retry", action="store_true", help="Actually retry with LLM (requires API key and PDF)")
    
    args = parser.parse_args()
    
    # Determine if input_path is a file or directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        # Single file mode
        output_dir = input_path.parent
        specific_file = str(input_path)
        logger.info(f"Processing specific file: {specific_file}")
    elif input_path.is_dir():
        # Directory mode
        output_dir = input_path
        specific_file = None
        logger.info(f"Processing directory: {output_dir}")
    else:
        print(f"❌ Error: {args.input_path} is not a valid file or directory")
        return
    
    # Initialize processor
    processor = FallbackRetryProcessor(str(output_dir), args.api_key, args.pdf_path)
    
    # Scan for fallbacks
    fallbacks = processor.scan_for_fallbacks(specific_file)
    
    if not fallbacks:
        print("✅ No fallback structures found! All content was processed successfully.")
        return
    
    # Analyze fallbacks
    analysis = processor.analyze_fallbacks()
    
    # Create retry batches
    batches = processor.extract_retry_batches(args.batch_size)
    
    # Generate retry script
    processor.generate_retry_script(batches)
    
    # Save analysis
    processor.save_retry_analysis(analysis, batches)
    
    # Print summary
    processor.print_summary(analysis, batches)
    
    # If retry flag is set and we have API key and PDF, actually retry
    if args.retry and args.api_key and args.pdf_path:
        print(f"\n🔄 Starting LLM retry processing...")
        retry_summary = await processor.process_retry_batches(batches)
        
        print(f"\n🎉 Retry processing complete!")
        print(f"📊 Success rate: {retry_summary['retry_summary']['success_rate']}")
        print(f"✅ Successful retries: {retry_summary['retry_summary']['successful_retries']}")
        print(f"❌ Failed retries: {retry_summary['retry_summary']['failed_retries']}")
        print(f"📁 Results saved to: {output_dir / 'retry_results.json'}")
    elif args.retry:
        print(f"\n⚠️  Retry requested but missing requirements:")
        if not args.api_key:
            print(f"   - Missing --api-key for LLM processing")
        if not args.pdf_path:
            print(f"   - Missing --pdf-path for content extraction")
        print(f"   Use --api-key and --pdf-path to enable actual retry processing")

if __name__ == "__main__":
    asyncio.run(main())
