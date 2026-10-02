"""The tools the harness can run, and the JSON that describes them to the model."""

import json
from pathlib import Path

from usda import USDAError, fetch_food, search_foods

DATA_DIR = Path(__file__).parent / "data"
FOODS = json.loads((DATA_DIR / "foods.json").read_text(encoding="utf-8"))["ingredients"]
SNAPSHOT = json.loads((DATA_DIR / "usda_snapshot.json").read_text(encoding="utf-8"))["foods"]

# Every known way of writing an ingredient -> its canonical name, e.g. "豆腐" -> "firm tofu".
ALIASES = {}
for canonical, entry in FOODS.items():
    ALIASES[canonical] = canonical
    for alias in entry["aliases"]:
        ALIASES[alias.lower()] = canonical

# Nutrition values do not change, so each food is fetched from USDA at most once per process.
nutrition_cache: dict[str, dict] = {}


def normalize_food_name(food_name: str) -> str:
    """Lowercase, collapse spaces, and map aliases: '  Tofu ' -> 'firm tofu'."""
    name = " ".join(food_name.lower().split())
    return ALIASES.get(name, name)


def get_food_nutrition(food_name: str) -> dict:
    """Per-100g nutrition for one food, as a dict. Raises USDAError if it cannot be found.

    Pinned foods are fetched by their exact FDC ID; anything else is searched by name.
    If USDA cannot be used at all, pinned foods fall back to the local snapshot.
    """
    name = normalize_food_name(food_name)
    if name in nutrition_cache:
        return nutrition_cache[name]

    from_snapshot = False
    try:
        if name in FOODS:
            food = fetch_food(FOODS[name]["fdc_id"])
            match_quality = "exact"
        else:
            matches = [f for f in search_foods(name) if f["per_100g"] is not None]
            if not matches:
                raise USDAError(
                    f"No USDA match for '{food_name}'.",
                    "Try a more generic English name for a single ingredient, e.g. 'pork, ground' "
                    "instead of a brand or dish name. For a composed dish, use estimate_dish_nutrition.",
                )
            food = matches[0]  # USDA ranks the best match first
            match_quality = "close"
    except USDAError as e:
        if not (e.unavailable and name in SNAPSHOT):
            raise
        food = SNAPSHOT[name]
        match_quality = "exact"
        from_snapshot = True

    source = "local snapshot of USDA data" if from_snapshot else "USDA FoodData Central"
    if food["data_type"] == "Branded":
        # One company's product label, not a USDA generic food. Say so.
        match_quality = "branded"
        source += ", branded product label"

    result = {
        "food_name": name,
        "matched_description": food["description"],
        "per_100g": food["per_100g"],
        "match_quality": match_quality,
        "source": source,
        "fdc_id": food["fdc_id"],
    }
    # Only cache live answers, so a snapshot fallback retries USDA next time.
    if not from_snapshot:
        nutrition_cache[name] = result
    return result


def lookup_food_nutrition(food_name: str) -> str:
    """Tool: calories, protein, carbs, and fat per 100g for one food or ingredient."""
    if not isinstance(food_name, str) or not food_name.strip():
        return json.dumps({
            "error": "food_name is empty.",
            "hint": "Pass the English name of one food or ingredient, e.g. 'firm tofu'.",
        })
    try:
        return json.dumps(get_food_nutrition(food_name), ensure_ascii=False)
    except USDAError as e:
        return json.dumps(e.to_dict(), ensure_ascii=False)


# What the model sees: the "set notes" in the screenplay.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_food_nutrition",
            "description": (
                "Look up calories, protein, carbohydrates, and fat per 100g for a single food or raw "
                "ingredient (e.g. 'tofu', 'cooked white rice', 'egg') from USDA FoodData Central. "
                "Use this for per-100g questions and comparisons between ingredients; call it once "
                "per food. Do NOT use it for a composed dish like ramen or bibimbap; use "
                "estimate_dish_nutrition for those. Values are per 100 g, not per serving. "
                "match_quality is 'exact' (a hand-picked USDA entry), 'close' (best search match, "
                "check matched_description), or 'branded' (a commercial product label)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "food_name": {
                        "type": "string",
                        "description": (
                            "Common English name of one food or ingredient, e.g. 'firm tofu', "
                            "'cooked wheat noodles', 'egg'. Translate non-English names to English "
                            "first (豆腐 -> 'tofu'). One food per call."
                        ),
                    },
                },
                "required": ["food_name"],
            },
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {"lookup_food_nutrition": lookup_food_nutrition}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({
            "error": f"Unknown tool '{name}'.",
            "hint": f"Call one of these tools instead: {list(TOOL_MAP)}.",
        })
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return json.dumps({
            "error": f"Bad arguments for {name}: {e}",
            "hint": "Check the tool's parameters and call again with only those arguments.",
        })
    except Exception as e:
        # A bug in a tool must not crash the chat. The message stays generic so no internals leak.
        return json.dumps({
            "error": f"{name} failed unexpectedly ({type(e).__name__}).",
            "hint": "Tell the user this lookup failed and do not guess the values.",
        })
