"""Tests for the tools.

These never call USDA: they use the local snapshot, or replace the USDA calls with stand-ins,
so they give the same result every run. The one live check at the bottom is marked `live`
and is skipped unless you run:  uv run pytest -m live
"""

import json
import os

import pytest

import tools
import usda

RESULT_KEYS = {"food_name", "matched_description", "per_100g", "match_quality", "source", "fdc_id"}
NUTRIENT_KEYS = {"calories_kcal", "protein_g", "carbs_g", "fat_g"}


@pytest.fixture(autouse=True)
def empty_cache():
    """Each test starts with nothing cached, so one test cannot answer for another."""
    tools.nutrition_cache.clear()


# --- Spec test 1: lookup_food_nutrition("tofu") returns the expected shape ---


def test_lookup_tofu_returns_expected_shape(monkeypatch):
    # Stand in for a live USDA answer using the recorded snapshot values.
    monkeypatch.setattr(tools, "fetch_food", lambda fdc_id: tools.SNAPSHOT["firm tofu"])

    result = json.loads(tools.lookup_food_nutrition("tofu"))

    assert set(result) == RESULT_KEYS
    assert set(result["per_100g"]) == NUTRIENT_KEYS
    assert result["food_name"] == "firm tofu"  # alias mapped to the canonical name
    assert result["fdc_id"] == 172448  # the pinned ID
    assert result["match_quality"] == "exact"
    assert result["source"] == "USDA FoodData Central"


def test_chinese_alias_maps_to_same_food(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)
    result = json.loads(tools.lookup_food_nutrition(" 豆腐 "))
    assert result["food_name"] == "firm tofu"


# --- Spec test 2: no USDA_API_KEY falls back to the snapshot and says so ---


def test_lookup_without_api_key_falls_back_to_snapshot(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)

    result = json.loads(tools.lookup_food_nutrition("egg"))

    assert result["source"] == "local snapshot of USDA data"
    assert result["per_100g"] == tools.SNAPSHOT["egg"]["per_100g"]
    assert "egg" not in tools.nutrition_cache  # USDA is retried next time


def test_without_api_key_unpinned_food_returns_error_with_hint(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)

    result = json.loads(tools.lookup_food_nutrition("durian"))

    assert result["error"] == "USDA_API_KEY is not set."
    assert result["hint"]


def test_branded_entry_is_labelled(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)

    result = json.loads(tools.lookup_food_nutrition("gochujang"))

    assert result["match_quality"] == "branded"
    assert result["source"] == "local snapshot of USDA data, branded product label"


# --- Spec test 3: a nonsense name returns an error with a hint ---


def test_lookup_nonsense_name_returns_error_with_hint(monkeypatch):
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: [])  # USDA found nothing

    result = json.loads(tools.lookup_food_nutrition("xyzzy blorp"))

    assert result["error"] == "No USDA match for 'xyzzy blorp'."
    assert result["hint"]


def test_empty_name_returns_error_with_hint():
    result = json.loads(tools.lookup_food_nutrition("   "))
    assert result["error"] and result["hint"]


# --- Energy fallback: pork belly (FDC 2727576) has no nutrient 1008 ---


def test_energy_falls_back_when_1008_is_missing(monkeypatch):
    # Same shape and values as USDA's /food/2727576 answer: energy only under 2047 and 2048,
    # and a slightly negative "carbohydrate by difference".
    detail = {
        "fdcId": 2727576,
        "description": "Pork, belly, with skin, raw",
        "dataType": "Foundation",
        "foodNutrients": [
            {"nutrient": {"id": 2047}, "amount": 380.0},
            {"nutrient": {"id": 2048}, "amount": 385.0},
            {"nutrient": {"id": 1003}, "amount": 15.2},
            {"nutrient": {"id": 1004}, "amount": 35.8},
            {"nutrient": {"id": 1005}, "amount": -0.705},
        ],
    }
    monkeypatch.setattr(usda, "_request", lambda method, path, json_body=None: detail)

    food = usda.fetch_food(2727576)

    # Atwater specific (2048) is used, not 0 and not the general factor (2047). Carbs clamp to 0.
    assert food["per_100g"] == {"calories_kcal": 385, "protein_g": 15.2, "carbs_g": 0.0, "fat_g": 35.8}


def test_pork_belly_snapshot_has_energy():
    assert tools.SNAPSHOT["pork belly"]["per_100g"]["calories_kcal"] == 385


# --- run_tool never crashes on a bad call ---


def test_run_tool_unknown_name_returns_error_with_hint():
    result = json.loads(tools.run_tool("get_weather", {"location": "NYC"}))
    assert result["error"] and result["hint"]


def test_run_tool_ignores_junk_before_the_tool_name():
    # Names Gemini actually sent during Phase 4 testing.
    for name in [" ROUILLER:default_api:estimate_portion_size", "لمات:estimate_portion_size"]:
        result = json.loads(tools.run_tool(name, {"food_name": "congee", "quantity": 1, "unit": "bowl"}))
        assert result["food_name"] == "congee"


def test_run_tool_bad_arguments_returns_error_with_hint():
    result = json.loads(tools.run_tool("lookup_food_nutrition", {"food": "tofu"}))
    assert result["error"] and result["hint"]


# --- Spec test 4: everyday portions through estimate_portion_size ---


def portion(food_name, quantity, unit, **kwargs):
    return json.loads(tools.estimate_portion_size(food_name, quantity, unit, **kwargs))


def rule_for(category, unit):
    return tools.PORTION_RULES["categories"][category]["units"][unit]


# Expected grams are computed from the rules file, so editing a number there does not break a test.
@pytest.mark.parametrize(
    "food_name, quantity, unit, category",
    [
        ("dumplings", 6, "piece", "dumpling"),
        ("xiaolongbao", 3, "piece", "xiaolongbao"),
        ("ramen", 0.5, "bowl", "noodle soup"),
        ("congee", 1, "bowl", "congee"),
        ("Korean fried chicken", 2, "piece", "fried chicken piece"),
    ],
)
def test_portions_use_the_right_rule(food_name, quantity, unit, category):
    rule = rule_for(category, unit)

    result = portion(food_name, quantity, unit)

    assert set(result) == {"food_name", "grams_low", "grams_typical", "grams_high", "assumption", "rule_source"}
    assert result["grams_low"] == round(quantity * rule["grams_low"])
    assert result["grams_typical"] == round(quantity * rule["grams_typical"])
    assert result["grams_high"] == round(quantity * rule["grams_high"])
    assert result["grams_low"] <= result["grams_typical"] <= result["grams_high"]
    assert result["rule_source"] == rule["source"]


def test_fried_chicken_assumption_says_edible_part():
    assert "without bone" in portion("Korean fried chicken", 2, "piece")["assumption"]


def test_size_changes_the_weight():
    small = portion("congee", 1, "bowl", size="small")
    regular = portion("congee", 1, "bowl")
    large = portion("congee", 1, "bowl", size="large")
    assert small["grams_typical"] < regular["grams_typical"] < large["grams_typical"]


def test_longest_name_wins_and_aliases_work():
    assert tools.find_portion_category("egg fried rice") == "fried rice"  # not plain rice
    assert tools.find_portion_category("rice") == "plain rice"
    assert tools.find_portion_category("soup dumplings") == "xiaolongbao"  # not soup, not dumpling
    assert tools.find_portion_category("pork dumplings") == "dumpling"
    assert tools.find_portion_category("粥") == "congee"
    assert tools.find_portion_category("japchae") is None


def test_unknown_food_uses_generic_rule_and_says_so():
    result = portion("japchae", 1, "plate")
    generic = rule_for("stir-fry dish", "plate")
    assert result["grams_typical"] == generic["grams_typical"]
    assert "No portion rule for 'japchae'" in result["assumption"]


# --- Spec test 5: an invalid unit for a food returns an error with a hint ---


def test_invalid_unit_for_food_returns_error_naming_valid_units():
    result = portion("kimchi jjigae", 1, "piece")
    assert result["error"] == "'piece' is not a valid unit for kimchi jjigae."
    assert "'bowl'" in result["hint"]


@pytest.mark.parametrize(
    "args",
    [
        ("congee", 1, "spoonful"),  # unit outside the enum
        ("congee", 0, "bowl"),  # nothing eaten
        ("congee", "half", "bowl"),  # not a number
        ("congee", 300, "bowl"),  # grams passed as a count
        ("", 1, "bowl"),  # no food
    ],
)
def test_bad_portion_arguments_return_error_with_hint(args):
    result = portion(*args)
    assert result["error"] and result["hint"]


def test_bad_size_returns_error_with_hint():
    result = portion("congee", 1, "bowl", size="huge")
    assert result["error"] and result["hint"]


# --- The data files agree with each other ---


def test_every_dish_has_a_recipe_and_a_portion_category():
    assert set(tools.DISHES) == set(tools.RECIPES)
    for dish, entry in tools.DISHES.items():
        assert entry["portion_category"] in tools.PORTION_RULES["categories"], dish


def test_recipes_only_use_pinned_ingredients():
    for dish, recipe in tools.RECIPES.items():
        for item in recipe["ingredients"] + [recipe["oil"]]:
            assert item["name"] in tools.FOODS, f"{dish}: '{item['name']}' is not pinned"
            assert item["name"] in tools.SNAPSHOT, f"{dish}: '{item['name']}' is not in the snapshot"


def test_reference_serving_matches_typical_portion():
    # A recipe served by the plate or bowl should weigh what the portion tool calls typical.
    for dish, recipe in tools.RECIPES.items():
        serving = recipe["reference_serving"]
        if serving["unit"] in ("plate", "bowl"):
            category = tools.DISHES[dish]["portion_category"]
            assert serving["grams"] == rule_for(category, serving["unit"])["grams_typical"], dish


def test_no_name_means_two_things():
    seen = {}
    for dish, entry in tools.DISHES.items():
        for name in [dish] + entry["aliases"]:
            assert name.lower() not in seen, f"'{name}' is used by {seen[name.lower()]} and {dish}"
            seen[name.lower()] = dish
    for category, rules in tools.PORTION_RULES["categories"].items():
        for name in rules["names"]:
            assert name not in seen, f"'{name}' is both a dish alias and a generic name in {category}"


# --- estimate_dish_nutrition (offline: pinned ingredients come from the snapshot) ---


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)


def dish(*args, **kwargs):
    return json.loads(tools.estimate_dish_nutrition(*args, **kwargs))


# Spec test 6: tomato scrambled eggs through the dish tool: low <= typical <= high.
def test_tomato_scrambled_eggs_range_is_ordered(offline):
    result = dish("tomato scrambled eggs", grams_low=200, grams_high=400)

    assert set(result) == {
        "dish_name", "grams_low", "grams_high", "calories", "protein_g", "carbs_g", "fat_g",
        "biggest_uncertainty", "confidence", "assumptions", "sources",
    }
    calories = result["calories"]
    assert 0 < calories["low"] <= calories["typical"] <= calories["high"]
    assert result["confidence"] == "template"
    assert result["biggest_uncertainty"] in ("portion_size", "cooking_oil")
    assert result["sources"] == ["local snapshot of USDA data"]


def test_exact_weight_gives_a_range_from_oil_only(offline):
    result = dish("stir-fried greens", grams_low=220)  # grams_high defaults to grams_low
    assert result["grams_low"] == result["grams_high"] == 220
    assert result["calories"]["low"] < result["calories"]["high"]
    assert result["biggest_uncertainty"] == "cooking_oil"
    assert any("bok choy" in a for a in result["assumptions"])


def test_dish_aliases_reach_the_template(offline):
    assert dish("小笼包", grams_low=100, grams_high=150)["dish_name"] == "xiaolongbao"
    assert dish("potstickers", grams_low=100, grams_high=150)["dish_name"] == "dumplings"


# Spec test 7: no template -> "no template" error; with an ingredients list -> an estimate.
def test_dish_without_template_errors_then_works_with_ingredients(offline):
    error = dish("japchae", grams_low=200, grams_high=300)
    assert error["error"] == "No built-in recipe for 'japchae'."
    assert "ingredients" in error["hint"] and "tomato scrambled eggs" in error["hint"]

    result = dish("japchae", grams_low=200, grams_high=300, ingredients=[
        {"name": "ground beef", "grams_low": 30, "grams_high": 50},
        {"name": "spinach", "grams_low": 30, "grams_high": 40},
        {"name": "carrot", "grams_low": 20, "grams_high": 30},
        {"name": "soy sauce", "grams_low": 10, "grams_high": 15},
        {"name": "sesame oil", "grams_low": 5, "grams_high": 10},
        {"name": "cooking oil", "grams_low": 5, "grams_high": 15},
    ])
    assert result["confidence"] == "ingredient_estimate"
    assert result["calories"]["low"] <= result["calories"]["typical"] <= result["calories"]["high"]


# Spec test 8: oil_level="light" gives a lower high end than "unknown".
def test_light_oil_lowers_the_high_end(offline):
    unknown = dish("tomato scrambled eggs", grams_low=200, grams_high=400)
    light = dish("tomato scrambled eggs", grams_low=200, grams_high=400, oil_level="light")
    heavy = dish("tomato scrambled eggs", grams_low=200, grams_high=400, oil_level="heavy")
    assert light["calories"]["high"] < unknown["calories"]["high"]
    assert heavy["calories"]["low"] > unknown["calories"]["low"]


def test_ramen_counts_half_the_broth(offline, monkeypatch):
    half = dish("ramen", grams_low=750, grams_high=750)
    assert any("half of the broth" in a for a in half["assumptions"])

    all_broth = dict(tools.RECIPES["ramen"], broth_eaten_fraction=1.0)
    monkeypatch.setitem(tools.RECIPES, "ramen", all_broth)
    whole = dish("ramen", grams_low=750, grams_high=750)
    assert whole["calories"]["typical"] > half["calories"]["typical"]


def test_close_match_is_listed_in_assumptions(offline, monkeypatch):
    fake = {"fdc_id": 1, "description": "Noodles, sweet potato, cooked", "data_type": "SR Legacy",
            "per_100g": {"calories_kcal": 100, "protein_g": 0.1, "carbs_g": 25.0, "fat_g": 0.0}}
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: [fake])

    result = dish("japchae", ingredients=[{"name": "sweet potato noodles", "grams_low": 100, "grams_high": 150}])

    assert "'sweet potato noodles' was matched to USDA 'Noodles, sweet potato, cooked' (close match)." in result["assumptions"]
    assert result["grams_low"] == 100  # no portion given: the sum of the ingredients

    # The sentence for the model to relay says it is lower confidence and names the loose match.
    assert "lower-confidence" in result["note_for_user"]
    assert "'sweet potato noodles' as USDA 'Noodles, sweet potato, cooked'" in result["note_for_user"]


def test_ingredient_weight_is_the_sum_and_mismatch_is_noted(offline):
    items = [{"name": "cooked white rice", "grams_low": 200, "grams_high": 300},
             {"name": "chicken breast", "grams_low": 100, "grams_high": 150}]  # total 300-450 g

    close = dish("chicken rice", grams_low=300, grams_high=400, ingredients=items)
    assert (close["grams_low"], close["grams_high"]) == (300, 450)  # the sum, not what was passed
    assert not any("differs from the ingredient total" in a for a in close["assumptions"])

    far = dish("chicken rice", grams_low=150, grams_high=250, ingredients=items)  # mid 200 vs 375
    assert (far["grams_low"], far["grams_high"]) == (300, 450)
    assert any("differs from the ingredient total" in a for a in far["assumptions"])


def test_template_result_has_no_note_for_user(offline):
    assert "note_for_user" not in dish("congee", grams_low=300, grams_high=300)


def test_single_pinned_food_needs_no_ingredient_list(offline):
    result = dish("rice", grams_low=150, grams_high=250)  # 'rice' is an alias of cooked white rice
    assert result["dish_name"] == "cooked white rice"
    rice = tools.SNAPSHOT["cooked white rice"]["per_100g"]["calories_kcal"]
    assert result["calories"]["low"] == round(150 * rice / 100)
    assert result["calories"]["high"] == round(250 * rice / 100)
    assert "note_for_user" not in result  # USDA data for one food, not a guessed recipe


def test_single_pinned_food_without_grams_returns_error(offline):
    result = dish("rice")
    assert result["error"] and "estimate_portion_size" in result["hint"]


# --- Search for unpinned foods ---


def fake_food(fdc_id, description):
    return {"fdc_id": fdc_id, "description": description, "data_type": "SR Legacy",
            "per_100g": {"calories_kcal": 100, "protein_g": 1.0, "carbs_g": 1.0, "fat_g": 1.0}}


def test_search_prefers_raw_and_names_starting_with_the_query(monkeypatch):
    results = [fake_food(1, "Fish oil, salmon"), fake_food(2, "Fish, salmon, chinook, raw")]
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: results)
    assert tools.search_best_match("salmon")["fdc_id"] == 2  # says "raw"

    results = [fake_food(1, "Fish, squid, raw"), fake_food(2, "Squid, dried"), fake_food(3, "Squid, raw")]
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: results)
    assert tools.search_best_match("squid")["fdc_id"] == 3  # starts with "squid" AND says "raw"

    # One point each: a tie, so USDA's order decides (the real squid results from USDA).
    results = [fake_food(1, "Mollusks, squid, mixed species, raw"), fake_food(2, "Squid (calamari), frozen, tubes only")]
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: results)
    assert tools.search_best_match("squid")["fdc_id"] == 1

    results = [fake_food(1, "Strawberry jam"), fake_food(2, "Strawberries, frozen")]
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: results)
    assert tools.search_best_match("jam")["fdc_id"] == 1  # "strawberry" does not count as "raw"


def test_search_uses_fndds_only_when_sr_and_foundation_find_nothing(monkeypatch):
    calls = []
    sr_and_foundation_results = []

    def fake_search(query, data_types, page_size):
        calls.append(data_types)
        if data_types == ["Survey (FNDDS)"]:
            return [fake_food(9, "Lomi salmon")]
        return sr_and_foundation_results

    monkeypatch.setattr(tools, "search_foods", fake_search)

    # SR Legacy and Foundation find nothing: FNDDS is searched next.
    assert tools.search_best_match("lomi salmon")["fdc_id"] == 9
    assert calls == [["SR Legacy", "Foundation"], ["Survey (FNDDS)"]]

    # SR Legacy and Foundation find something: FNDDS is never searched.
    calls.clear()
    sr_and_foundation_results.append(fake_food(1, "Fish, salmon, raw"))
    assert tools.search_best_match("salmon")["fdc_id"] == 1
    assert calls == [["SR Legacy", "Foundation"]]


def test_sushi_piece_rule():
    result = portion("salmon nigiri", 6, "piece")
    rule = rule_for("sushi piece", "piece")
    assert result["grams_low"] == 6 * rule["grams_low"]
    assert result["grams_high"] == 6 * rule["grams_high"]


def test_branded_ingredient_is_listed_in_assumptions(offline):
    result = dish("bibimbap", grams_low=500, grams_high=500)
    assert any("branded product label" in a for a in result["assumptions"])


def test_unknown_ingredient_returns_error_with_hint(offline, monkeypatch):
    monkeypatch.setattr(tools, "search_foods", lambda query, **kwargs: [])
    result = dish("mapo tofu", ingredients=[{"name": "doubanjiang", "grams_low": 10, "grams_high": 20}])
    assert result["error"] == "Could not find nutrition data for ingredient 'doubanjiang'."
    assert "closer generic ingredient" in result["hint"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dish_name": "ramen"},  # built-in dish but no portion
        {"dish_name": "ramen", "grams_low": 0},
        {"dish_name": "ramen", "grams_low": 500, "grams_high": 400},
        {"dish_name": "ramen", "grams_low": 500, "oil_level": "very"},
        {"dish_name": ""},
        {"dish_name": "japchae", "ingredients": [{"name": "egg"}]},  # no grams
        {"dish_name": "japchae", "ingredients": "egg, noodles"},  # not a list
    ],
)
def test_bad_dish_arguments_return_error_with_hint(offline, kwargs):
    result = dish(**kwargs)
    assert result["error"] and result["hint"]


# --- Live check: calls the real USDA API. Run with: uv run pytest -m live ---


@pytest.mark.live
def test_live_usda_lookup():
    if not os.environ.get("USDA_API_KEY"):
        pytest.skip("USDA_API_KEY is not set")

    pinned = json.loads(tools.lookup_food_nutrition("tofu"))
    assert pinned["source"] == "USDA FoodData Central"
    assert pinned["fdc_id"] == 172448
    assert pinned["per_100g"] == tools.SNAPSHOT["firm tofu"]["per_100g"]

    searched = json.loads(tools.lookup_food_nutrition("squid"))  # not pinned
    assert searched["match_quality"] == "close"
    assert set(searched) == RESULT_KEYS
    assert searched["matched_description"] == "Mollusks, squid, mixed species, raw"  # SR Legacy, raw


# --- Dry flour listed at the cooked weight ---

TAKOYAKI_FLOUR = {"name": "wheat flour", "grams_low": 70, "grams_high": 130}
TAKOYAKI_REST = [
    {"name": "egg", "grams_low": 20, "grams_high": 40},
    {"name": "scallion", "grams_low": 5, "grams_high": 10},
    {"name": "cooking oil", "grams_low": 5, "grams_high": 15},
]


def test_flour_heavy_list_without_water_returns_error_with_hint(offline):
    result = dish("takoyaki", ingredients=[TAKOYAKI_FLOUR, *TAKOYAKI_REST])
    assert result["error"]
    assert "Flour and starch make up 68% of this dish's weight" in result["hint"]
    assert "'water' item" in result["hint"]


def test_same_weight_with_water_passes_with_fewer_calories(offline):
    # Same flour weight as above, but half of it is now the water the batter holds.
    with_water = [{"name": "wheat flour", "grams_low": 35, "grams_high": 65},
                  {"name": "water", "grams_low": 35, "grams_high": 65}, *TAKOYAKI_REST]
    result = dish("takoyaki", ingredients=with_water)
    assert (result["grams_low"], result["grams_high"]) == (100, 195)  # same total as the flour-only list

    all_flour = [{"name": "wheat flour", "grams_low": 70, "grams_high": 130}, *TAKOYAKI_REST]
    flour_kcal = sum(
        (i["grams_low"] + i["grams_high"]) / 2 * tools.SNAPSHOT[i["name"]]["per_100g"]["calories_kcal"] / 100
        for i in all_flour
    )
    assert result["calories"]["typical"] < flour_kcal


def test_breaded_dish_with_a_little_starch_and_no_water_passes(offline):
    result = dish("fried chicken", ingredients=[
        {"name": "chicken breast", "grams_low": 150, "grams_high": 200},
        {"name": "cornstarch", "grams_low": 15, "grams_high": 25},
        {"name": "wheat flour", "grams_low": 10, "grams_high": 15},
        {"name": "cooking oil", "grams_low": 10, "grams_high": 20},
    ])
    assert "error" not in result


def test_fried_dough_mostly_flour_with_a_little_water_passes(offline):
    result = dish("youtiao", ingredients=[
        {"name": "wheat flour", "grams_low": 40, "grams_high": 60},
        {"name": "cooking oil", "grams_low": 10, "grams_high": 20},
        {"name": "water", "grams_low": 5, "grams_high": 10},
    ])
    assert "error" not in result
    assert result["calories"]["low"] > 0


def test_water_adds_grams_but_no_calories_and_no_usda_call(offline, monkeypatch):
    looked_up = []
    real = tools.get_food_nutrition
    monkeypatch.setattr(tools, "get_food_nutrition", lambda name: looked_up.append(name) or real(name))
    egg = {"name": "egg", "grams_low": 100, "grams_high": 100}

    plain = dish("steamed egg", ingredients=[egg])
    watered = dish("steamed egg", ingredients=[egg, {"name": "Water", "grams_low": 50, "grams_high": 50}])

    assert watered["grams_low"] == plain["grams_low"] + 50
    assert watered["calories"] == plain["calories"]
    assert (watered["protein_g"], watered["carbs_g"], watered["fat_g"]) == (plain["protein_g"], plain["carbs_g"], plain["fat_g"])
    assert looked_up == ["egg", "egg"]  # water was never looked up
