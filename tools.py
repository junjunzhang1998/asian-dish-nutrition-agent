"""The tools the harness can run, and the JSON that describes them to the model."""

import json
from pathlib import Path

from usda import USDAError, fetch_food, search_foods

DATA_DIR = Path(__file__).parent / "data"
FOODS_FILE = json.loads((DATA_DIR / "foods.json").read_text(encoding="utf-8"))
FOODS = FOODS_FILE["ingredients"]
DISHES = FOODS_FILE["dishes"]
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


# --- estimate_portion_size ---

PORTION_RULES = json.loads((DATA_DIR / "portion_rules.json").read_text(encoding="utf-8"))
UNITS = ["piece", "bowl", "plate", "cup"]
SIZES = ["small", "regular", "large"]
MAX_QUANTITY = 50  # more units than anyone eats; a bigger number is probably grams


def error_json(error: str, hint: str) -> str:
    return json.dumps({"error": error, "hint": hint}, ensure_ascii=False)


# Every name the portion tool recognizes -> its portion category. Dish names and their aliases
# come from foods.json; generic words that are not a built-in dish ('soup', 'udon') come from
# portion_rules.json.
PORTION_NAMES = {}
for category, rules in PORTION_RULES["categories"].items():
    for name in rules["names"]:
        PORTION_NAMES[name] = category
for dish, entry in DISHES.items():
    for name in [dish] + entry["aliases"]:
        PORTION_NAMES[name.lower()] = entry["portion_category"]


def find_portion_category(food: str) -> str | None:
    """Return the portion category a food belongs to, or None if no name matches.

    A name matches if the food equals it or contains it as whole words. The longest
    match wins, so 'kimchi fried rice' is fried rice, not plain rice.
    """
    best_category, best_name = None, ""
    for name, category in PORTION_NAMES.items():
        if f" {name} " in f" {food} " and len(name) > len(best_name):
            best_category, best_name = category, name
    return best_category


def estimate_portion_size(food_name: str, quantity: float, unit: str, size: str = "regular") -> str:
    """Tool: turn an everyday portion ('3 dumplings', 'half a bowl of ramen') into a gram range."""
    if not isinstance(food_name, str) or not food_name.strip():
        return error_json("food_name is empty.", "Pass the English name of the food, e.g. 'xiaolongbao' or 'ramen'.")
    # bool is checked separately because Python counts True and False as numbers.
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or quantity <= 0:
        return error_json(
            f"quantity must be a number greater than 0, got {quantity!r}.",
            "Pass how many units were eaten as a number: 3 for three pieces, 0.5 for half a bowl.",
        )
    if quantity > MAX_QUANTITY:
        return error_json(
            f"quantity {quantity:g} is too large to be a number of {unit}s.",
            "quantity counts units (pieces, bowls), not grams. If the user gave a weight in grams, "
            "skip this tool and use that weight directly.",
        )
    if unit not in UNITS:
        return error_json(f"'{unit}' is not a valid unit.", f"unit must be one of {UNITS}.")
    if size not in SIZES:
        return error_json(f"'{size}' is not a valid size.", f"size must be one of {SIZES}, or leave it out for 'regular'.")

    food = " ".join(food_name.lower().split())
    category = find_portion_category(food)
    known_food = category is not None
    if not known_food:
        category = PORTION_RULES["generic_category_by_unit"][unit]
    units = PORTION_RULES["categories"][category]["units"]

    if unit not in units:
        valid = " or ".join(f"'{u}'" for u in units)
        return error_json(
            f"'{unit}' is not a valid unit for {food}.",
            f"Use unit {valid} for {food} (portion category: {category}).",
        )

    rule = units[unit]
    multiplier = PORTION_RULES["size_multipliers"][size]
    low_each = rule["grams_low"] * multiplier
    typical_each = rule["grams_typical"] * multiplier
    high_each = rule["grams_high"] * multiplier

    # Build the sentence that explains the estimate, e.g. "3 pieces at 20-30 g each."
    size_word = "" if size == "regular" else f"{size} "
    plural = "s" if quantity > 1 else ""
    assumption = f"{quantity:g} {size_word}{unit}{plural} at {round(low_each)}-{round(high_each)} g each."
    if size != "regular":
        assumption += f" A {size} {unit} is taken as {multiplier} x a regular one."
    if not known_food:
        assumption += f" No portion rule for '{food}', so the generic rule for a {unit} ({category}) was used."
    if "note" in rule:
        assumption += " " + rule["note"]

    return json.dumps({
        "food_name": food,
        "grams_low": round(quantity * low_each),
        "grams_typical": round(quantity * typical_each),
        "grams_high": round(quantity * high_each),
        "assumption": assumption,
        "rule_source": rule["source"],
    }, ensure_ascii=False)


# --- estimate_dish_nutrition ---

RECIPES = json.loads((DATA_DIR / "recipes.json").read_text(encoding="utf-8"))["recipes"]
OIL_LEVELS = ["light", "normal", "heavy", "unknown"]

# Added fats. In an ingredients list, these are narrowed by oil_level and count as oil
# when deciding whether oil or portion size drives the calorie range.
OIL_INGREDIENTS = {"cooking oil", "sesame oil", "pork fat"}

# Every known way of writing a dish -> its canonical name, e.g. "小笼包" -> "xiaolongbao".
DISH_ALIASES = {}
for dish, entry in DISHES.items():
    DISH_ALIASES[dish] = dish
    for alias in entry["aliases"]:
        DISH_ALIASES[alias.lower()] = dish

NO_INGREDIENT_HINT = (
    "Replace it with a closer generic ingredient (e.g. 'chili bean paste' or 'soy sauce') "
    "or drop it if the amount is small, then call again."
)


def is_number(value) -> bool:
    # bool is excluded because Python counts True and False as numbers.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def narrow_oil(grams_low: float, grams_high: float, oil_level: str) -> tuple[float, float]:
    """Keep the third of the oil range the user described. 'unknown' keeps all of it."""
    third = (grams_high - grams_low) / 3
    if oil_level == "light":
        return grams_low, grams_low + third
    if oil_level == "normal":
        return grams_low + third, grams_high - third
    if oil_level == "heavy":
        return grams_high - third, grams_high
    return grams_low, grams_high


# A "line" is one ingredient with the grams actually eaten: low, typical, and high.
# "oil_spread" is how many of those grams of range come from oil alone (0 for food).


def template_lines(recipe: dict, grams_low: float, grams_high: float, oil_level: str) -> tuple[list, list]:
    """Scale a recipe from its reference serving to the portion eaten. Returns (lines, assumptions)."""
    serving = recipe["reference_serving"]
    scale_low = grams_low / serving["grams"]
    scale_high = grams_high / serving["grams"]
    scale_mid = (scale_low + scale_high) / 2
    broth_eaten = recipe.get("broth_eaten_fraction", 1.0)

    lines = []
    for item in recipe["ingredients"]:
        grams = item["grams"] * (broth_eaten if item.get("in_broth") else 1.0)
        lines.append({
            "name": item["name"],
            "low": grams * scale_low,
            "typical": grams * scale_mid,
            "high": grams * scale_high,
            "oil_spread": 0.0,
        })

    # low = low portion with low oil; high = high portion with high oil; typical = both midpoints.
    oil = recipe["oil"]
    oil_low, oil_high = narrow_oil(oil["grams_low"], oil["grams_high"], oil_level)
    oil_eaten = broth_eaten if oil.get("in_broth") else 1.0
    lines.append({
        "name": oil["name"],
        "low": oil_low * oil_eaten * scale_low,
        "typical": (oil_low + oil_high) / 2 * oil_eaten * scale_mid,
        "high": oil_high * oil_eaten * scale_high,
        "oil_spread": (oil_high - oil_low) * oil_eaten * scale_mid,  # the oil range at the typical portion
    })

    unit = serving["unit"]
    assumptions = [f"1 reference {unit} = {serving['grams']} g as served; scaled to the {grams_low:g}-{grams_high:g} g eaten."]
    if oil["grams_high"] == 0:
        assumptions.append("No added oil.")
    else:
        assumptions.append(
            f"{oil['name'].capitalize()}: {oil_low:.0f}-{oil_high:.0f} g per reference {unit} (oil_level: {oil_level})."
        )
    return lines, assumptions + recipe["assumptions"]


def parse_ingredients(ingredients) -> tuple[list, str | None]:
    """Check an ingredients list from the model. Returns (items, None) or ([], error JSON)."""
    hint = (
        "Pass `ingredients` as a list of objects like "
        "{'name': 'egg', 'grams_low': 50, 'grams_high': 100}, with grams for the amount eaten."
    )
    if not isinstance(ingredients, list):
        return [], error_json("ingredients must be a list.", hint)

    items = []
    for item in ingredients:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            return [], error_json(f"Ingredient {item!r} has no name.", hint)
        low = item.get("grams_low")
        high = item.get("grams_high", low)  # an exact amount can leave out grams_high
        if not (is_number(low) and is_number(high)) or low < 0 or high < low:
            return [], error_json(
                f"Ingredient '{item['name']}' needs grams_low and grams_high with 0 <= grams_low <= grams_high.",
                hint,
            )
        items.append({"name": item["name"], "grams_low": low, "grams_high": high})
    return items, None


def ingredient_lines(items: list, oil_level: str) -> list:
    """Turn the model's ingredient list into lines. Only the fats are narrowed by oil_level."""
    lines = []
    for item in items:
        low, high = item["grams_low"], item["grams_high"]
        is_oil = normalize_food_name(item["name"]) in OIL_INGREDIENTS
        if is_oil:
            low, high = narrow_oil(low, high, oil_level)
        lines.append({
            "name": item["name"],
            "low": low,
            "typical": (low + high) / 2,
            "high": high,
            "oil_spread": high - low if is_oil else 0.0,
        })
    return lines


def add_up(lines: list) -> tuple[dict, list, list]:
    """Look up every line per 100 g and add up calories and macros.

    Returns (totals, assumptions about the matches, sources). Raises USDAError.
    """
    totals = {"low": 0.0, "typical": 0.0, "high": 0.0, "oil_spread": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    assumptions, sources = [], []

    for line in lines:
        try:
            food = get_food_nutrition(line["name"])  # the same helper as lookup_food_nutrition
        except USDAError as e:
            if e.unavailable:
                raise
            raise USDAError(f"Could not find nutrition data for ingredient '{line['name']}'.", NO_INGREDIENT_HINT)

        per_gram = {key: value / 100 for key, value in food["per_100g"].items()}
        for key in ("low", "typical", "high", "oil_spread"):
            totals[key] += line[key] * per_gram["calories_kcal"]
        for key in ("protein_g", "carbs_g", "fat_g"):  # macros are for the typical portion
            totals[key] += line["typical"] * per_gram[key]

        if food["match_quality"] == "close":
            assumptions.append(f"'{line['name']}' was matched to USDA '{food['matched_description']}' (close match).")
        elif food["match_quality"] == "branded":
            assumptions.append(f"'{line['name']}' uses a branded product label: '{food['matched_description']}'.")
        if food["source"] not in sources:
            sources.append(food["source"])

    return totals, assumptions, sources


def estimate_dish_nutrition(
    dish_name: str,
    grams_low: float | None = None,
    grams_high: float | None = None,
    ingredients: list | None = None,
    oil_level: str = "unknown",
) -> str:
    """Tool: calories (low / typical / high) and macros for a portion of a dish, from its ingredients."""
    if not isinstance(dish_name, str) or not dish_name.strip():
        return error_json("dish_name is empty.", "Pass the English dish name, e.g. 'tomato scrambled eggs'.")
    if oil_level not in OIL_LEVELS:
        return error_json(f"'{oil_level}' is not a valid oil_level.", f"oil_level must be one of {OIL_LEVELS}.")
    if grams_low is not None:
        if grams_high is None:
            grams_high = grams_low  # an exact weight
        if not (is_number(grams_low) and is_number(grams_high)) or grams_low <= 0 or grams_high < grams_low:
            return error_json(
                f"Invalid portion: grams_low={grams_low!r}, grams_high={grams_high!r}.",
                "grams_low and grams_high must be numbers with 0 < grams_low <= grams_high. "
                "Get them from estimate_portion_size, or use the user's exact weight for both.",
            )

    dish = " ".join(dish_name.lower().split())

    # 1. The model (or the user) gave the ingredients: calculate from those.
    if ingredients:
        items, error = parse_ingredients(ingredients)
        if error:
            return error
        lines = ingredient_lines(items, oil_level)
        confidence = "ingredient_estimate"
        assumptions = ["Calculated from the ingredient list given, not a built-in recipe."]
        if grams_low is None:
            grams_low = sum(item["grams_low"] for item in items)
            grams_high = sum(item["grams_high"] for item in items)
            assumptions.append("Portion weight is the sum of the listed ingredients.")

    # 2. A built-in recipe: scale it to the portion eaten.
    elif dish in DISH_ALIASES:
        if grams_low is None:
            return error_json(
                f"No portion weight given for '{dish_name}'.",
                "Call estimate_portion_size first and pass its grams_low and grams_high, "
                "or pass the user's exact weight as both.",
            )
        dish = DISH_ALIASES[dish]
        lines, assumptions = template_lines(RECIPES[dish], grams_low, grams_high, oil_level)
        confidence = "template"

    # 3. Neither: ask the model for an ingredient list.
    else:
        return error_json(
            f"No built-in recipe for '{dish_name}'.",
            "Call again with an `ingredients` list: your best estimate of each ingredient and its gram "
            "range for the portion eaten, including cooking oil. Built-in recipes: " + ", ".join(RECIPES) + ".",
        )

    try:
        totals, match_notes, sources = add_up(lines)
    except USDAError as e:
        return json.dumps(e.to_dict(), ensure_ascii=False)

    # Which assumption drives the range: the oil's own spread, or everything else (the portion)?
    oil_part = totals["oil_spread"]
    portion_part = (totals["high"] - totals["low"]) - oil_part
    biggest_uncertainty = "cooking_oil" if oil_part > portion_part else "portion_size"

    return json.dumps({
        "dish_name": dish,
        "grams_low": round(grams_low),
        "grams_high": round(grams_high),
        "calories": {"low": round(totals["low"]), "typical": round(totals["typical"]), "high": round(totals["high"])},
        "protein_g": round(totals["protein_g"]),
        "carbs_g": round(totals["carbs_g"]),
        "fat_g": round(totals["fat_g"]),
        "biggest_uncertainty": biggest_uncertainty,
        "confidence": confidence,
        "assumptions": assumptions + match_notes,
        "sources": sources,
    }, ensure_ascii=False)


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
    {
        "type": "function",
        "function": {
            "name": "estimate_portion_size",
            "description": (
                "Convert an everyday portion of an East Asian food ('3 dumplings', 'half a bowl of "
                "ramen', '2 pieces of Korean fried chicken', '半碗粥') into an estimated weight in "
                "grams, as a low/typical/high range. Call this whenever the user describes a portion "
                "without giving grams; call it once per food. Skip it when the user gives an exact "
                "weight. It returns weight only, not calories. The result's `assumption` says what "
                "one unit was taken to weigh; for bone-in foods the grams are the edible part."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "food_name": {
                        "type": "string",
                        "description": (
                            "Canonical English dish or food name, e.g. 'xiaolongbao', 'ramen', "
                            "'congee', 'Korean fried chicken'. Translate non-English names to "
                            "English first (粥 -> 'congee')."
                        ),
                    },
                    "quantity": {
                        "type": "number",
                        "description": (
                            "How many units were eaten, e.g. 3 for three pieces. Use 0.5 for 'half' "
                            "or 半, and 1 for 'a' or 一. This is a count of units, never grams."
                        ),
                    },
                    "unit": {
                        "type": "string",
                        "enum": UNITS,
                        "description": (
                            "'piece' (个: one dumpling, one bao, one wing or drumstick), "
                            "'bowl' (碗: soups, noodle soups, stews, congee, rice), "
                            "'plate' (盘 / 份: one restaurant serving of a dish), "
                            "'cup' (杯: soup or rice)."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "enum": SIZES,
                        "description": (
                            "'small' (小), 'regular', or 'large' (大). Leave it out unless the user "
                            "says the portion was small or large; the default is 'regular'."
                        ),
                    },
                },
                "required": ["food_name", "quantity", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_dish_nutrition",
            "description": (
                "Estimate calories, protein, carbohydrates, and fat for a portion of an East Asian dish "
                "by calculating from its ingredients. Returns a low-to-high calorie range and says "
                "whether portion size or cooking oil drives the uncertainty. Has built-in recipes for "
                "common dishes: " + ", ".join(RECIPES) + ". For a built-in dish, pass grams_low and "
                "grams_high (from estimate_portion_size, or the user's exact weight). For any other "
                "dish, or when the user tells you what went into it, pass an `ingredients` list. "
                "Use this for dishes and portions; use lookup_food_nutrition only for per-100g "
                "questions about a single ingredient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {
                        "type": "string",
                        "description": (
                            "Canonical English dish name, e.g. 'tomato scrambled eggs', 'xiaolongbao', "
                            "'japchae'. Translate non-English names first (番茄炒蛋 -> 'tomato scrambled eggs')."
                        ),
                    },
                    "grams_low": {
                        "type": "number",
                        "description": (
                            "Low estimate of the portion eaten, in grams, e.g. 200. From "
                            "estimate_portion_size, or the user's exact weight. Required when no "
                            "`ingredients` are given."
                        ),
                    },
                    "grams_high": {
                        "type": "number",
                        "description": "High estimate of the portion eaten, in grams, e.g. 300. Equal to grams_low when the weight is exact.",
                    },
                    "ingredients": {
                        "type": "array",
                        "description": (
                            "Use when the dish has no built-in recipe or the user described the "
                            "ingredients. One item per ingredient (never combine two ingredients in one "
                            "item), with grams for the amount actually eaten, including cooking oil, e.g. "
                            "[{'name': 'egg', 'grams_low': 50, 'grams_high': 100}, {'name': 'cooking oil', "
                            "'grams_low': 5, 'grams_high': 15}]. Prefer these names, which have exact USDA "
                            "data: " + ", ".join(FOODS) + ". Any other name is searched in USDA and may "
                            "match loosely, so use a plain single-ingredient name like 'glass noodles' "
                            "or 'shiitake mushrooms'."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Generic English ingredient name, e.g. 'ground pork', 'cooked white rice'."},
                                "grams_low": {"type": "number", "description": "Low estimate of grams eaten."},
                                "grams_high": {"type": "number", "description": "High estimate of grams eaten."},
                            },
                            "required": ["name", "grams_low", "grams_high"],
                        },
                    },
                    "oil_level": {
                        "type": "string",
                        "enum": OIL_LEVELS,
                        "description": (
                            "How oily the dish was: 'light', 'normal', 'heavy', or 'unknown' (default). "
                            "Set it only when the user says how oily it was, e.g. 'pretty oily' -> 'heavy'."
                        ),
                    },
                },
                "required": ["dish_name"],
            },
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "lookup_food_nutrition": lookup_food_nutrition,
    "estimate_portion_size": estimate_portion_size,
    "estimate_dish_nutrition": estimate_dish_nutrition,
}


def clean_tool_name(name: str) -> str:
    """Gemini sometimes sends a name with junk in front, e.g. 'x:default_api:estimate_portion_size'.

    The real name is the part after the last colon.
    """
    return name.split(":")[-1].strip()


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    name = clean_tool_name(name)
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
