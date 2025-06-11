import fitz
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
from datetime import datetime
import re
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IslamicContentSchema:
    """Schema for different types of Islamic scholarly content"""
    
    # Content types with their descriptions
    CONTENT_TYPES = {
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

@dataclass
class ParsingMemory:
    """Memory system to track parsing patterns and improve accuracy"""
    
    # Pattern recognition memory
    heading_patterns: Dict[str, float] = field(default_factory=dict)
    question_patterns: Dict[str, float] = field(default_factory=dict)
    arabic_patterns: Set[str] = field(default_factory=set)
    scholar_names: Set[str] = field(default_factory=set)
    
    # Structural memory
    document_structure: List[Dict] = field(default_factory=list)
    section_hierarchy: Dict[int, str] = field(default_factory=dict)
    page_contexts: Dict[int, Dict] = field(default_factory=dict)
    
    # Content classification memory
    classification_history: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    confidence_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Cross-reference memory
    quran_references: Dict[str, List[int]] = field(default_factory=dict)
    hadith_references: Dict[str, List[int]] = field(default_factory=dict)
    topic_continuity: Dict[str, List[int]] = field(default_factory=dict)
    
    # Error correction memory
    correction_patterns: Dict[str, str] = field(default_factory=dict)
    llm_feedback: List[Dict] = field(default_factory=list)
    
    def add_classification_memory(self, text: str, content_type: str, confidence: float):
        """Record classification decisions for pattern learning"""
        # Extract features from text for pattern matching
        features = self._extract_text_features(text)
        
        for feature in features:
            self.classification_history[feature][content_type] += 1
        
        self.confidence_scores[content_type].append(confidence)
    
    def _extract_text_features(self, text: str) -> List[str]:
        """Extract features from text for pattern recognition"""
        features = []
        
        # Length-based features
        if len(text) < 50:
            features.append("short_text")
        elif len(text) > 200:
            features.append("long_text")
        
        # Pattern-based features
        if re.match(r'^\d+\.\s', text):
            features.append("numbered_item")
        if re.match(r'^Q\d*[:\.]', text, re.IGNORECASE):
            features.append("question_marker")
        if re.match(r'^A\d*[:\.]', text, re.IGNORECASE):
            features.append("answer_marker")
        if re.search(r'[\u0600-\u06FF]', text):  # Arabic characters
            features.append("contains_arabic")
        if text.isupper() and len(text) < 100:
            features.append("all_caps")
        if re.search(r'\b(Question|Answer|Fatwa|Ruling)\b', text, re.IGNORECASE):
            features.append("islamic_keyword")
        
        return features
    
    def predict_content_type(self, text: str) -> Optional[str]:
        """Predict content type based on learned patterns"""
        features = self._extract_text_features(text)
        
        type_scores = defaultdict(float)
        
        for feature in features:
            if feature in self.classification_history:
                for content_type, count in self.classification_history[feature].items():
                    type_scores[content_type] += count
        
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 2:  # Threshold for confidence
                return best_type[0]
        
        return None
    
    def add_structural_context(self, page_num: int, section_info: Dict):
        """Track document structure for context"""
        self.page_contexts[page_num] = section_info
        self.document_structure.append({
            "page": page_num,
            "section_type": section_info.get("type"),
            "heading": section_info.get("heading"),
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context_for_page(self, page_num: int) -> Dict:
        """Get contextual information for a page"""
        context = {
            "current_section": None,
            "recent_patterns": [],
            "expected_continuation": None
        }
        
        # Find current section context
        for p in range(max(1, page_num - 5), page_num + 1):
            if p in self.page_contexts:
                page_ctx = self.page_contexts[p]
                if page_ctx.get("section_type") in ["heading_major", "heading_minor"]:
                    context["current_section"] = page_ctx
                    break
        
        # Analyze recent patterns
        recent_structure = self.document_structure[-10:]  # Last 10 entries
        pattern_counts = Counter(item["section_type"] for item in recent_structure if item["section_type"])
        context["recent_patterns"] = pattern_counts.most_common(3)
        
        return context

class AnthropicLLMParser:
    def __init__(self, pdf_path: str, api_key: str, batch_size: int = 20):
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.batch_size = batch_size
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # Memory system
        self.memory = ParsingMemory()
        
        # Output directories
        self.output_dir = Path("llm_parsed_content")
        self.output_dir.mkdir(exist_ok=True)
        
        # Schema for prompt
        self.schema = IslamicContentSchema()
        
        # Load existing memory if available
        self.load_memory()
        
        logger.info(f"Initialized LLM parser with memory for {self.total_pages} pages")

    def load_memory(self):
        """Load existing memory from previous runs"""
        memory_file = self.output_dir / "parsing_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                
                # Restore memory components
                self.memory.heading_patterns = memory_data.get("heading_patterns", {})
                self.memory.question_patterns = memory_data.get("question_patterns", {})
                self.memory.arabic_patterns = set(memory_data.get("arabic_patterns", []))
                self.memory.scholar_names = set(memory_data.get("scholar_names", []))
                self.memory.classification_history = defaultdict(lambda: defaultdict(int), memory_data.get("classification_history", {}))
                self.memory.quran_references = memory_data.get("quran_references", {})
                self.memory.hadith_references = memory_data.get("hadith_references", {})
                
                logger.info("Loaded existing parsing memory")
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")

    def save_memory(self):
        """Save current memory state"""
        memory_data = {
            "heading_patterns": self.memory.heading_patterns,
            "question_patterns": self.memory.question_patterns,
            "arabic_patterns": list(self.memory.arabic_patterns),
            "scholar_names": list(self.memory.scholar_names),
            "classification_history": dict(self.memory.classification_history),
            "quran_references": self.memory.quran_references,
            "hadith_references": self.memory.hadith_references,
            "document_structure": self.memory.document_structure[-100:],  # Keep last 100 entries
            "saved_at": datetime.now().isoformat()
        }
        
        memory_file = self.output_dir / "parsing_memory.json"
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

    def extract_page_content(self, page_num: int) -> Dict[str, Any]:
        """Extract raw text content from a page with basic metadata"""
        try:
            page = self.doc[page_num]
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
                    
                    # Memory-based pre-classification
                    predicted_type = self.memory.predict_content_type(line_text)
                    
                    content_blocks.append({
                        "text": line_text,
                        "font": primary_font,
                        "size": max_font_size,
                        "bbox": line_bbox,
                        "is_centered": is_centered,
                        "is_bold": "Bold" in primary_font,
                        "y_position": line_bbox[1],
                        "predicted_type": predicted_type  # Memory prediction
                    })
            
            # Sort by Y position (top to bottom)
            content_blocks.sort(key=lambda x: x["y_position"])
            
            return {
                "page_number": page_num + 1,
                "content_blocks": content_blocks,
                "page_width": page_width,
                "page_height": page.rect.height
            }
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {e}")
            return {"page_number": page_num + 1, "content_blocks": [], "page_width": 0, "page_height": 0}

    def create_analysis_prompt(self, page_contents: List[Dict]) -> str:
        """Create a structured prompt for LLM analysis with memory context"""
        
        # Get context from memory
        page_numbers = [p["page_number"] for p in page_contents]
        context = self.memory.get_context_for_page(page_numbers[0])
        
        # Combine text from multiple pages for context
        combined_text = ""
        memory_predictions = []
        
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
                
                # Add memory prediction if available
                if block.get("predicted_type"):
                    formatting_hints.append(f"MEMORY_SUGGESTS:{block['predicted_type']}")
                    memory_predictions.append(f"'{block['text'][:50]}...' -> {block['predicted_type']}")
                
                hint_str = f" [{', '.join(formatting_hints)}]" if formatting_hints else ""
                combined_text += f"{block['text']}{hint_str}\n"
        
        content_types_desc = "\n".join([f"- {k}: {v}" for k, v in self.schema.CONTENT_TYPES.items()])
        
        # Build context information
        context_info = ""
        if context["current_section"]:
            context_info += f"CURRENT SECTION CONTEXT: {context['current_section']['heading']} ({context['current_section']['section_type']})\n"
        
        if context["recent_patterns"]:
            patterns_str = ", ".join([f"{pattern}: {count}" for pattern, count in context["recent_patterns"]])
            context_info += f"RECENT PATTERNS: {patterns_str}\n"
        
        if memory_predictions:
            context_info += f"MEMORY PREDICTIONS: {'; '.join(memory_predictions[:5])}\n"
        
        prompt = f"""
You are analyzing Islamic scholarly text from "Majmoo'al-Fatawa of Ibn Bazz". Your task is to identify and classify different types of content according to their semantic meaning and structure.

{context_info}

CONTENT TYPES TO IDENTIFY:
{content_types_desc}

TEXT TO ANALYZE:
{combined_text}

INSTRUCTIONS:
1. Consider the MEMORY_SUGGESTS hints which are based on learned patterns from previous pages
2. Use the CURRENT SECTION CONTEXT to maintain consistency
3. Pay attention to RECENT PATTERNS to understand document flow
4. Identify ALL content pieces and classify them using the content types above
5. Pay special attention to:
   - CENTERED and BOLD text are often headings
   - "Foreword", "Introduction", "Chapter" are major headings
   - Questions usually start with numbers or "Question:"
   - Arabic text vs transliteration vs translation
   - Signature lines and attributions
   - Quranic verses and Hadith citations

6. Group related content under appropriate headings
7. Preserve the logical flow and hierarchy
8. Include confidence scores for your classifications

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
          "notes": "any additional context",
          "memory_agreement": true/false
        }}
      ]
    }}
  ],
  "metadata": {{
    "pages_analyzed": [list of page numbers],
    "total_sections_found": number,
    "primary_language": "english|arabic|mixed",
    "context_used": true/false,
    "memory_predictions_considered": number
  }}
}}

CRITICAL: Return ONLY the JSON response, no additional text or explanation.
"""
        return prompt

    async def call_anthropic_api(self, session: aiohttp.ClientSession, prompt: str) -> Dict[str, Any]:
        """Make API call to Anthropic Claude"""
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
        
        try:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["content"][0]["text"]
                    
                    # Try to parse JSON from response
                    try:
                        parsed_result = json.loads(content)
                        
                        # Update memory with LLM results
                        self.update_memory_from_llm_result(parsed_result)
                        
                        return parsed_result
                    except json.JSONDecodeError:
                        # Extract JSON if wrapped in other text
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx != -1 and end_idx != 0:
                            parsed_result = json.loads(content[start_idx:end_idx])
                            self.update_memory_from_llm_result(parsed_result)
                            return parsed_result
                        else:
                            logger.error(f"Failed to parse JSON from response: {content[:200]}...")
                            return {"error": "JSON parsing failed", "raw_response": content}
                else:
                    error_text = await response.text()
                    logger.error(f"API call failed: {response.status} - {error_text}")
                    return {"error": f"API error: {response.status}", "details": error_text}
                    
        except Exception as e:
            logger.error(f"Exception during API call: {e}")
            return {"error": "Exception", "details": str(e)}

    def update_memory_from_llm_result(self, result: Dict[str, Any]):
        """Update memory based on LLM classification results"""
        if "sections" not in result:
            return
        
        for section in result["sections"]:
            # Update structural memory
            if "heading" in section:
                heading = section["heading"]
                page_num = heading.get("page_number", 0)
                self.memory.add_structural_context(page_num, {
                    "type": heading.get("content_type"),
                    "heading": heading.get("text"),
                    "confidence": heading.get("confidence", 0)
                })
            
            # Update classification memory
            for item in section.get("content_items", []):
                self.memory.add_classification_memory(
                    item["text"],
                    item["content_type"],
                    item.get("confidence", 0.5)
                )
                
                # Track specific patterns
                if item["content_type"] == "quran_verse":
                    self.track_quran_reference(item["text"], item["page_number"])
                elif item["content_type"] == "hadith":
                    self.track_hadith_reference(item["text"], item["page_number"])

    def track_quran_reference(self, text: str, page_num: int):
        """Track Quran verse references for cross-referencing"""
        # Simple pattern matching for Quranic references
        quran_pattern = r'\b(\d+):(\d+)\b'  # Surah:Ayah pattern
        matches = re.findall(quran_pattern, text)
        
        for surah, ayah in matches:
            ref_key = f"{surah}:{ayah}"
            if ref_key not in self.memory.quran_references:
                self.memory.quran_references[ref_key] = []
            self.memory.quran_references[ref_key].append(page_num)

    def track_hadith_reference(self, text: str, page_num: int):
        """Track Hadith references for cross-referencing"""
        # Track common hadith collections
        hadith_collections = ["Bukhari", "Muslim", "Tirmidhi", "Abu Dawud", "Nasa'i", "Ibn Majah"]
        
        for collection in hadith_collections:
            if collection.lower() in text.lower():
                if collection not in self.memory.hadith_references:
                    self.memory.hadith_references[collection] = []
                self.memory.hadith_references[collection].append(page_num)

    async def process_batch_with_llm(self, batch_pages: List[Dict]) -> Dict[str, Any]:
        """Process a batch of pages with LLM analysis"""
        prompt = self.create_analysis_prompt(batch_pages)
        
        async with aiohttp.ClientSession() as session:
            result = await self.call_anthropic_api(session, prompt)
            
            if "error" in result:
                logger.error(f"LLM analysis failed: {result}")
                return self.create_fallback_structure(batch_pages)
            
            return result

    def create_fallback_structure(self, batch_pages: List[Dict]) -> Dict[str, Any]:
        """Create fallback structure if LLM fails, using memory predictions"""
        sections = []
        
        for page_data in batch_pages:
            page_num = page_data["page_number"]
            content_items = []
            
            for block in page_data["content_blocks"]:
                # Use memory prediction if available, otherwise use heuristics
                if block.get("predicted_type"):
                    content_type = block["predicted_type"]
                    confidence = 0.7  # Memory-based prediction
                    notes = "memory_prediction"
                else:
                    # Fallback heuristics
                    if block["is_bold"] and block["is_centered"]:
                        content_type = "heading_major"
                    elif "Question" in block["text"] or block["text"].startswith(("Q:", "A:")):
                        content_type = "question" if "Question" in block["text"] else "answer"
                    elif any(arabic_char in block["text"] for arabic_char in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"):
                        content_type = "arabic_text"
                    else:
                        content_type = "paragraph"
                    
                    confidence = 0.5
                    notes = "fallback_classification"
                
                content_items.append({
                    "text": block["text"],
                    "content_type": content_type,
                    "page_number": page_num,
                    "confidence": confidence,
                    "notes": notes
                })
                
                # Still update memory with fallback classifications
                self.memory.add_classification_memory(block["text"], content_type, confidence)
            
            if content_items:
                sections.append({
                    "type": "content_section",
                    "heading": {
                        "text": f"Page {page_num} Content",
                        "content_type": "heading_minor",
                        "page_number": page_num,
                        "confidence": 0.5
                    },
                    "content_items": content_items
                })
        
        return {
            "sections": sections,
            "metadata": {
                "pages_analyzed": [p["page_number"] for p in batch_pages],
                "total_sections_found": len(sections),
                "primary_language": "mixed",
                "processing_method": "fallback_with_memory"
            }
        }

    async def process_document_in_batches(self) -> Dict[str, Any]:
        """Process the entire document in batches with LLM analysis and memory"""
        logger.info(f"Starting memory-enhanced LLM processing of {self.total_pages} pages")
        
        all_sections = []
        processing_stats = {
            "total_pages": self.total_pages,
            "batches_processed": 0,
            "successful_llm_calls": 0,
            "failed_llm_calls": 0,
            "memory_predictions_used": 0,
            "total_sections": 0,
            "start_time": datetime.now().isoformat(),
            "processing_time_seconds": 0
        }
        
        start_time = time.time()
        
        # Process in batches
        for start_page in range(0, self.total_pages, self.batch_size):
            end_page = min(start_page + self.batch_size, self.total_pages)
            logger.info(f"Processing batch: pages {start_page + 1} to {end_page}")
            
            # Extract content from batch
            batch_pages = []
            for page_num in range(start_page, end_page):
                page_content = self.extract_page_content(page_num)
                if page_content["content_blocks"]:  # Only add pages with content
                    batch_pages.append(page_content)
                    
                    # Count memory predictions
                    memory_preds = sum(1 for block in page_content["content_blocks"] 
                                     if block.get("predicted_type"))
                    processing_stats["memory_predictions_used"] += memory_preds
            
            if not batch_pages:
                continue
            
            # Analyze with LLM
            try:
                batch_result = await self.process_batch_with_llm(batch_pages)
                
                if "error" not in batch_result:
                    processing_stats["successful_llm_calls"] += 1
                    all_sections.extend(batch_result.get("sections", []))
                else:
                    processing_stats["failed_llm_calls"] += 1
                    logger.warning(f"Batch processing failed, using fallback")
                
            except Exception as e:
                logger.error(f"Error processing batch {start_page//self.batch_size + 1}: {e}")
                processing_stats["failed_llm_calls"] += 1
            
            processing_stats["batches_processed"] += 1
            
            # Save memory state periodically
            if processing_stats["batches_processed"] % 5 == 0:
                self.save_memory()
            
            # Rate limiting - be respectful to API
            await asyncio.sleep(1)
            
            # Save intermediate results every 10 batches
            if processing_stats["batches_processed"] % 10 == 0:
                await self.save_intermediate_results(all_sections, processing_stats["batches_processed"])
        
        # Final memory save
        self.save_memory()
        
        processing_stats["total_sections"] = len(all_sections)
        processing_stats["processing_time_seconds"] = time.time() - start_time
        processing_stats["end_time"] = datetime.now().isoformat()
        
        # Create final result
        final_result = {
            "document_info": {
                "title": "Majmoo'al-Fatawa of Ibn Bazz",
                "total_pages": self.total_pages,
                "total_sections": len(all_sections),
                "processing_method": "llm_powered_with_memory",
                "processing_stats": processing_stats,
                "memory_stats": self.get_memory_stats()
            },
            "sections": all_sections
        }
        
        # Save final results
        await self.save_final_results(final_result)
        
        return final_result

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage and effectiveness"""
        return {
            "total_patterns_learned": len(self.memory.classification_history),
            "quran_references_tracked": len(self.memory.quran_references),
            "hadith_references_tracked": len(self.memory.hadith_references),
            "document_structure_entries": len(self.memory.document_structure),
            "unique_scholar_names": len(self.memory.scholar_names),
            "arabic_patterns": len(self.memory.arabic_patterns)
        }

    # [Rest of the methods remain the same as original...]
    async def save_intermediate_results(self, sections: List[Dict], batch_num: int):
        """Save intermediate results"""
        filename = self.output_dir / f"intermediate_batch_{batch_num}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "batch_number": batch_num,
                "sections_count": len(sections),
                "timestamp": datetime.now().isoformat(),
                "sections": sections[-self.batch_size * 5:]  # Save last 5 batches worth
            }, f, indent=2, ensure_ascii=False)

    async def save_final_results(self, result: Dict[str, Any]):
        """Save final structured results"""
        
        # Complete structured result
        with open(self.output_dir / "llm_parsed_complete.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # React-friendly structure
        react_structure = self.convert_to_react_structure(result)
        with open(self.output_dir / "react_islamic_components.json", 'w', encoding='utf-8') as f:
            json.dump(react_structure, f, indent=2, ensure_ascii=False)
        
        # Headings index for navigation
        headings_index = self.create_headings_index(result)
        with open(self.output_dir / "headings_index.json", 'w', encoding='utf-8') as f:
            json.dump(headings_index, f, indent=2, ensure_ascii=False)
        
        # Processing summary
        summary = self.create_processing_summary(result)
        with open(self.output_dir / "processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Memory analysis report
        memory_report = self.create_memory_report()
        with open(self.output_dir / "memory_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(memory_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Memory-Enhanced LLM Processing Complete!")
        logger.info(f"📁 Results saved to: {self.output_dir}")
        logger.info(f"📊 Total sections: {len(result['sections'])}")
        logger.info(f"🧠 Memory predictions used: {result['document_info']['processing_stats']['memory_predictions_used']}")
        logger.info(f"⏱️  Processing time: {result['document_info']['processing_stats']['processing_time_seconds']:.2f} seconds")
        logger.info(f"🤖 Successful LLM calls: {result['document_info']['processing_stats']['successful_llm_calls']}")
        logger.info(f"❌ Failed LLM calls: {result['document_info']['processing_stats']['failed_llm_calls']}")

    def create_memory_report(self) -> Dict[str, Any]:
        """Create detailed memory analysis report"""
        
        # Analyze classification patterns
        pattern_analysis = {}
        for feature, classifications in self.memory.classification_history.items():
            if sum(classifications.values()) > 3:  # Only patterns with enough data
                pattern_analysis[feature] = {
                    "total_occurrences": sum(classifications.values()),
                    "most_common_type": max(classifications.items(), key=lambda x: x[1]),
                    "confidence": max(classifications.values()) / sum(classifications.values()),
                    "distribution": dict(classifications)
                }
        
        # Analyze confidence trends
        confidence_analysis = {}
        for content_type, scores in self.memory.confidence_scores.items():
            if scores:
                confidence_analysis[content_type] = {
                    "average_confidence": sum(scores) / len(scores),
                    "min_confidence": min(scores),
                    "max_confidence": max(scores),
                    "total_classifications": len(scores),
                    "improving_trend": len(scores) > 5 and scores[-3:] > scores[:3] if len(scores) > 6 else None
                }
        
        # Cross-reference analysis
        cross_ref_stats = {
            "quran_verses": {
                "unique_references": len(self.memory.quran_references),
                "total_occurrences": sum(len(pages) for pages in self.memory.quran_references.values()),
                "most_cited": sorted(self.memory.quran_references.items(), 
                                   key=lambda x: len(x[1]), reverse=True)[:10]
            },
            "hadith_collections": {
                "collections_found": len(self.memory.hadith_references),
                "total_references": sum(len(pages) for pages in self.memory.hadith_references.values()),
                "most_referenced": sorted(self.memory.hadith_references.items(),
                                        key=lambda x: len(x[1]), reverse=True)[:5]
            }
        }
        
        return {
            "memory_effectiveness": {
                "patterns_learned": len(pattern_analysis),
                "reliable_patterns": len([p for p in pattern_analysis.values() if p["confidence"] > 0.7]),
                "classification_accuracy_estimate": sum(p["confidence"] for p in pattern_analysis.values()) / len(pattern_analysis) if pattern_analysis else 0
            },
            "pattern_analysis": pattern_analysis,
            "confidence_analysis": confidence_analysis,
            "cross_reference_analysis": cross_ref_stats,
            "document_structure_insights": {
                "total_structural_points": len(self.memory.document_structure),
                "section_types_identified": list(set(item["section_type"] for item in self.memory.document_structure if item["section_type"])),
                "structural_consistency": self.analyze_structural_consistency()
            },
            "recommendations": self.generate_memory_recommendations()
        }

    def analyze_structural_consistency(self) -> Dict[str, Any]:
        """Analyze consistency in document structure"""
        if not self.memory.document_structure:
            return {"status": "no_data"}
        
        # Analyze section type transitions
        transitions = []
        for i in range(1, len(self.memory.document_structure)):
            prev_type = self.memory.document_structure[i-1]["section_type"]
            curr_type = self.memory.document_structure[i]["section_type"]
            if prev_type and curr_type:
                transitions.append(f"{prev_type} -> {curr_type}")
        
        transition_counts = Counter(transitions)
        
        return {
            "total_transitions": len(transitions),
            "unique_patterns": len(transition_counts),
            "most_common_transitions": transition_counts.most_common(5),
            "consistency_score": len(transition_counts.most_common(3)) / max(1, len(transition_counts))  # How concentrated are the patterns
        }

    def generate_memory_recommendations(self) -> List[str]:
        """Generate recommendations for improving memory usage"""
        recommendations = []
        
        # Check pattern reliability
        reliable_patterns = 0
        total_patterns = len(self.memory.classification_history)
        
        for classifications in self.memory.classification_history.values():
            if sum(classifications.values()) > 5:  # Enough samples
                max_count = max(classifications.values())
                total_count = sum(classifications.values())
                if max_count / total_count > 0.8:  # High confidence
                    reliable_patterns += 1
        
        if total_patterns > 0 and reliable_patterns / total_patterns < 0.5:
            recommendations.append("Consider processing more pages to improve pattern reliability")
        
        # Check cross-reference coverage
        if len(self.memory.quran_references) < 10:
            recommendations.append("Limited Quranic references detected - verify Arabic text recognition")
        
        if len(self.memory.hadith_references) < 3:
            recommendations.append("Few Hadith collections identified - may need better pattern matching")
        
        # Check confidence trends
        low_confidence_types = []
        for content_type, scores in self.memory.confidence_scores.items():
            if scores and sum(scores) / len(scores) < 0.6:
                low_confidence_types.append(content_type)
        
        if low_confidence_types:
            recommendations.append(f"Low confidence in classifying: {', '.join(low_confidence_types[:3])}")
        
        if not recommendations:
            recommendations.append("Memory system performing well - continue current approach")
        
        return recommendations

    def convert_to_react_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM results to React component structure"""
        react_sections = []
        
        for section in result["sections"]:
            react_section = {
                "type": "IslamicContentSection",
                "props": {
                    "heading": section["heading"],
                    "sectionType": section.get("type", "content_section"),
                    "memoryEnhanced": True
                },
                "children": []
            }
            
            for content_item in section.get("content_items", []):
                component_type = self.map_to_react_component(content_item["content_type"])
                
                child = {
                    "type": component_type,
                    "props": {
                        "content": content_item["text"],
                        "contentType": content_item["content_type"],
                        "pageNumber": content_item["page_number"],
                        "confidence": content_item.get("confidence", 1.0),
                        "notes": content_item.get("notes", ""),
                        "memoryAgreement": content_item.get("memory_agreement", None)
                    }
                }
                react_section["children"].append(child)
            
            react_sections.append(react_section)
        
        return {
            "document": result["document_info"],
            "sections": react_sections,
            "memoryStats": result["document_info"].get("memory_stats", {})
        }

    def map_to_react_component(self, content_type: str) -> str:
        """Map content types to React component names"""
        mapping = {
            "heading_major": "MajorHeading",
            "heading_minor": "MinorHeading", 
            "heading_question": "QuestionHeading",
            "quran_verse": "QuranVerse",
            "hadith": "HadithText",
            "fatwa_ruling": "FatwaRuling",
            "question": "Question",
            "answer": "Answer",
            "arabic_text": "ArabicText",
            "transliteration": "Transliteration",
            "translation": "Translation",
            "signature": "Signature",
            "citation": "Citation",
            "footnote": "Footnote",
            "paragraph": "Paragraph",
            "list_item": "ListItem",
            "navigation": "Navigation"
        }
        return mapping.get(content_type, "GenericContent")

    def create_headings_index(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create an index of all headings for navigation"""
        headings = []
        
        for i, section in enumerate(result["sections"]):
            heading = section.get("heading", {})
            headings.append({
                "section_id": i,
                "text": heading.get("text", ""),
                "type": heading.get("content_type", ""),
                "page_number": heading.get("page_number", 0),
                "confidence": heading.get("confidence", 0)
            })
        
        return {
            "total_headings": len(headings),
            "headings": headings,
            "by_type": self.group_headings_by_type(headings),
            "memory_enhanced": True
        }

    def group_headings_by_type(self, headings: List[Dict]) -> Dict[str, List[Dict]]:
        """Group headings by their type"""
        grouped = {}
        for heading in headings:
            heading_type = heading["type"]
            if heading_type not in grouped:
                grouped[heading_type] = []
            grouped[heading_type].append(heading)
        return grouped

    def create_processing_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the processing results"""
        stats = result["document_info"]["processing_stats"]
        memory_stats = result["document_info"]["memory_stats"]
        
        content_type_counts = {}
        for section in result["sections"]:
            for item in section.get("content_items", []):
                content_type = item["content_type"]
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        return {
            "processing_summary": {
                "total_pages": stats["total_pages"],
                "total_sections": stats["total_sections"],
                "processing_time": f"{stats['processing_time_seconds']:.2f} seconds",
                "success_rate": f"{(stats['successful_llm_calls'] / max(1, stats['successful_llm_calls'] + stats['failed_llm_calls']) * 100):.1f}%",
                "memory_predictions_used": stats["memory_predictions_used"],
                "memory_enhancement": "enabled"
            },
            "memory_impact": {
                "patterns_learned": memory_stats["total_patterns_learned"],
                "cross_references_tracked": memory_stats["quran_references_tracked"] + memory_stats["hadith_references_tracked"],
                "structural_insights": memory_stats["document_structure_entries"],
                "memory_effectiveness": "high" if memory_stats["total_patterns_learned"] > 50 else "moderate"
            },
            "content_analysis": {
                "content_type_distribution": content_type_counts,
                "most_common_types": sorted(content_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "files_generated": [
                "llm_parsed_complete.json - Full structured content with memory",
                "react_islamic_components.json - React component structure", 
                "headings_index.json - Navigation index",
                "processing_summary.json - This summary",
                "memory_analysis.json - Memory system analysis",
                "parsing_memory.json - Learned patterns for future use"
            ]
        }

# Usage function
async def main(pdf_path: str, api_key: str, batch_size: int = 2):
    """Main function to process PDF with memory-enhanced LLM"""
    parser = AnthropicLLMParser(pdf_path, api_key, batch_size)
    result = await parser.process_document_in_batches()
    
    print(f"\n🎉 Memory-Enhanced LLM Processing Complete!")
    print(f"📄 Processed: {result['document_info']['total_pages']} pages")
    print(f"📑 Created: {result['document_info']['total_sections']} sections")
    print(f"🧠 Memory predictions: {result['document_info']['processing_stats']['memory_predictions_used']}")
    print(f"⏱️  Time taken: {result['document_info']['processing_stats']['processing_time_seconds']:.2f} seconds")
    print(f"🤖 LLM Success Rate: {(result['document_info']['processing_stats']['successful_llm_calls'] / max(1, result['document_info']['processing_stats']['batches_processed']) * 100):.1f}%")
    print(f"📊 Patterns learned: {result['document_info']['memory_stats']['total_patterns_learned']}")
    print(f"📁 Check the 'llm_parsed_content' folder for results")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <pdf_path> <anthropic_api_key> [batch_size]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    api_key = sys.argv[2] 
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    # Run the async function
    asyncio.run(main(pdf_path, api_key, batch_size))