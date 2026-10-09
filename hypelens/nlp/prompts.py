"""Prompt engineering templates for Gemma 4 multimodal extraction.

Guides Gemma 4 to extract structured financial claims, event categories,
rumour vs confirmed status, aspect sentiment, and manipulation flags.
"""

from __future__ import annotations

from typing import Optional


GEMMA_EXTRACTION_SYSTEM_PROMPT = """You are HypeLens, an expert financial intelligence analyst extracting structured claims from multimodal community posts (text, OCR from charts/reels, video transcripts).

You must analyze the content and return ONLY a valid JSON object matching this schema:
{
  "ticker": "STOCK_TICKER or null",
  "claim": "Concise summary of the core market claim",
  "event_category": "earnings | merger_acquisition | product_launch | regulatory | management_change | general_speculation | other",
  "status": "rumour_unverified | credible_report | officially_confirmed | denied_retracted",
  "sentiment": number between -1.0 (extremely bearish) and 1.0 (extremely bullish),
  "intensity": number between 0.0 (weak/mild) and 1.0 (extreme excitement or panic),
  "aspects": {
    "growth": number between -1.0 and 1.0,
    "valuation": number between -1.0 and 1.0,
    "product": number between -1.0 and 1.0
  },
  "is_promotional_spam": boolean,
  "confidence": number between 0.0 and 1.0
}

Rules:
1. Do not invent facts or tickers that are not mentioned or implied.
2. Unverified rumors must be labeled "rumour_unverified" with appropriate confidence.
3. Flag obvious promotional pumping, bot repetitions, or rocket emoji spam as is_promotional_spam: true.
4. Return ONLY raw JSON without preamble, thoughts, or markdown fences.
"""


def build_extraction_prompt(
    content_text: str,
    ocr_text: Optional[str] = None,
    transcript: Optional[str] = None,
    has_images: bool = False,
) -> str:
    """Construct user prompt for Gemma 4 multimodal analysis.

    Args:
        content_text: Main post or caption text.
        ocr_text: Extracted text from on-screen images or charts.
        transcript: Audio transcript from Reels or videos.
        has_images: Whether image frames or charts are attached to the prompt.

    Returns:
        Formatted prompt string.
    """
    sections = [
        "Analyze the following community financial content and output the required JSON schema:\n",
        f"--- POST TEXT ---\n{content_text.strip() or '[No text]'}\n",
    ]

    if ocr_text:
        sections.append(f"--- ON-SCREEN OCR TEXT ---\n{ocr_text.strip()}\n")

    if transcript:
        sections.append(f"--- SPOKEN TRANSCRIPT ---\n{transcript.strip()}\n")

    if has_images:
        sections.append("--- ATTACHED VISUALS ---\n[Refer to the attached image frames / chart screenshots]\n")

    sections.append(
        "Return the JSON object now. Remember: output valid JSON only."
    )

    return "\n".join(sections)
