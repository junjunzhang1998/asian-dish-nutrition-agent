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
    monkeypatch.setattr(tools, "search_foods", lambda query: [])  # USDA found nothing

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


def test_run_tool_bad_arguments_returns_error_with_hint():
    result = json.loads(tools.run_tool("lookup_food_nutrition", {"food": "tofu"}))
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

    searched = json.loads(tools.lookup_food_nutrition("broccoli"))  # not pinned
    assert searched["match_quality"] == "close"
    assert set(searched) == RESULT_KEYS
