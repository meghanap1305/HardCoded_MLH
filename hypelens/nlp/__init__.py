"""NLP and cognitive extraction modules for HypeLens."""

from hypelens.nlp.extractor import (
    ExtractionResult,
    GemmaExtractor,
    encode_image_to_base64,
)
from hypelens.nlp.parser import (
    EpistemicStatus,
    EventCategory,
    ExtractedClaim,
    clean_and_extract_json,
    parse_gemma_extraction,
)
from hypelens.nlp.prompts import (
    GEMMA_EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)

__all__ = [
    "EpistemicStatus",
    "EventCategory",
    "ExtractedClaim",
    "clean_and_extract_json",
    "parse_gemma_extraction",
    "GEMMA_EXTRACTION_SYSTEM_PROMPT",
    "build_extraction_prompt",
    "GemmaExtractor",
    "ExtractionResult",
    "encode_image_to_base64",
]
