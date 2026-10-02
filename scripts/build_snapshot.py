"""Fetch every pinned ingredient from USDA and save the values to data/usda_snapshot.json.

One-off helper, not used by the app. The snapshot is the fallback the tool uses when USDA
is unreachable or USDA_API_KEY is missing. Re-run it after changing a pinned FDC ID.
Run from the repo root:  uv run python scripts/build_snapshot.py
"""

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import usda` works when run as a script

from usda import USDAError, fetch_food  # noqa: E402


def main() -> None:
    foods = json.loads((ROOT / "data" / "foods.json").read_text(encoding="utf-8"))["ingredients"]

    snapshot = {}
    for name, entry in foods.items():
        try:
            snapshot[name] = fetch_food(entry["fdc_id"])
        except USDAError as e:
            # Stop rather than write a partial snapshot.
            sys.exit(f"{name} (FDC {entry['fdc_id']}): {e.error}")
        print(f"{name:<24} {snapshot[name]['per_100g']}")

    out = {
        "_about": f"Per-100g values for the pinned ingredients in data/foods.json, fetched from USDA FoodData Central on {date.today().isoformat()} by scripts/build_snapshot.py. Fallback only; the tool calls USDA first.",
        "foods": snapshot,
    }
    path = ROOT / "data" / "usda_snapshot.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(snapshot)} foods to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
