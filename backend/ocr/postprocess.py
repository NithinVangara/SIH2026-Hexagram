import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize OCR text without attempting to correct its meaning.

    This function:
    - normalizes Unicode representation
    - replaces repeated whitespace
    - removes unnecessary leading/trailing whitespace

    It does NOT:
    - guess incorrect OCR characters
    - extract declarations
    - alter numbers
    - convert currencies
    - infer missing text
    """
    if not isinstance(text, str):
        raise TypeError("OCR text must be a string")

    # Normalize Unicode into a consistent representation.
    text = unicodedata.normalize("NFKC", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text_blocks(text_blocks: list[dict]) -> list[dict]:
    """
    Normalize OCR text while preserving all evidence metadata.

    The original OCR text is preserved under `raw_text`.
    """
    normalized_blocks = []

    for block in text_blocks:
        normalized_block = dict(block)

        raw_text = block["text"]
        normalized_block["raw_text"] = raw_text
        normalized_block["text"] = normalize_text(raw_text)

        normalized_blocks.append(normalized_block)

    return normalized_blocks