from engine import OCREngine
from postprocess import normalize_text_blocks

def main():
    engine = OCREngine()

    result = engine.process("/Users/nithinvangara/Desktop/SIH2026-Hexagram/data/samples/product.png")
    result["text_blocks"] = normalize_text_blocks(
    result["text_blocks"]
    )

    print("\nOCR RESULT")
    print("=" * 50)

    for block in result["text_blocks"]:
        print(f"ID:          {block['id']}")
        print(f"Raw text:    {block['raw_text']}")
        print(f"Text:        {block['text']}")
        print(f"Confidence:  {block['confidence']:.4f}")
        print(f"BBox:        {block['bbox']}")
        print(f"Polygon:     {block['polygon']}")
        print("-" * 50)

    print("Orientation:", result["orientation"])


if __name__ == "__main__":
    main()