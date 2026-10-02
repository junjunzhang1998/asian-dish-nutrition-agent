"""Print the top USDA candidates for each ingredient in data/foods.json, to pick FDC IDs to pin.

One-off helper, not used by the app. Needs USDA_API_KEY in the environment.
Run from the repo root:  uv run python scripts/find_fdc_candidates.py

The suggested pick is the first SR Legacy entry in the top 3. SR Legacy is USDA's frozen
reference dataset: its IDs never change, its entries are generic foods, and it always
reports kcal directly. The suggestion is a starting point; the final choice is made by hand.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import usda` works when run as a script

from usda import USDAError, search_foods  # noqa: E402

TOP_N = 3


def suggest(candidates: list[dict]) -> tuple[dict | None, str]:
    """Return (suggested candidate, one-line reason)."""
    for food in candidates:
        if food["data_type"] == "SR Legacy" and food["per_100g"] is not None:
            return food, "first SR Legacy match: stable ID, generic food, kcal reported directly"
    for food in candidates:
        if food["per_100g"] is not None:
            return food, "no SR Legacy match in the top 3; first entry with complete data"
    return None, "no candidate has complete data; pick by hand"


def show(food: dict, mark: str) -> None:
    n = food["per_100g"]
    kcal = f"{n['calories_kcal']} kcal" if n else "kcal n/a"
    print(f"  {mark} {food['fdc_id']:>8}  {food['data_type']:<15} {kcal:>9}  {food['description']}")


def main() -> None:
    foods = json.loads((ROOT / "data" / "foods.json").read_text(encoding="utf-8"))["ingredients"]

    for name, entry in foods.items():
        print(f"\n{name}   (search: '{entry['usda_query']}')")
        try:
            candidates = search_foods(entry["usda_query"], page_size=TOP_N)
            note = ""
            if not candidates:
                # Nothing generic exists (e.g. gochujang). Show branded products instead, clearly labelled.
                candidates = search_foods(entry["usda_query"], data_types=["Branded"], page_size=TOP_N)
                note = "  NOTE: no generic USDA entry; these are BRANDED products"
        except USDAError as e:
            print(f"  ERROR: {e.error}")
            continue

        if note:
            print(note)
        if not candidates:
            print("  no matches at all")
            continue

        pick, reason = suggest(candidates)
        for food in candidates:
            show(food, "->" if food is pick else "  ")
        print(f"     suggested: {pick['fdc_id'] if pick else 'none'} ({reason})")


if __name__ == "__main__":
    main()
