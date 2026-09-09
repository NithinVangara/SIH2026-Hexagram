from backend.extraction.price import extract_mrp
from backend.extraction.quantity import extract_quantity
from backend.extraction.dates import extract_dates


def extract_declarations(text_blocks: list[dict]) -> dict:
    """
    Consolidate deterministic declaration extraction.

    M1 extracts observable package information only.
    No legal/compliance decision is made here.
    """

    mrp = extract_mrp(text_blocks)
    quantity = extract_quantity(text_blocks)
    dates = extract_dates(text_blocks)

    return {
        "mrp": mrp,
        "net_quantity": quantity,
        "manufacturing_date": dates["manufacturing_date"],
        "packing_date": dates["packing_date"],
        "expiry_date": dates["expiry_date"],
        "best_before": dates["best_before"],
    }