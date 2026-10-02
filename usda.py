"""USDA FoodData Central client: search foods by name, or fetch one food by its FDC ID.

Both calls return the same small dict, never the raw USDA response:
    {"fdc_id": 168878, "description": "...", "data_type": "SR Legacy",
     "per_100g": {"calories_kcal": 130, "protein_g": 2.7, "carbs_g": 28.2, "fat_g": 0.3}}

Every failure raises USDAError, which carries an "error" and a "hint" for the model.
"""

import os

import requests

BASE_URL = "https://api.nal.usda.gov/fdc/v1"
TIMEOUT_SECONDS = 5

# Generic foods only. "Branded" entries are one company's product label.
GENERIC_DATA_TYPES = ["Foundation", "SR Legacy", "Survey (FNDDS)"]

# USDA nutrient IDs. Some Foundation foods have no 1008 and only report the Atwater
# energy values, so energy is read from the first of these that is present.
ENERGY_KCAL_IDS = [1008, 2048, 2047]  # kcal, Atwater specific (kcal), Atwater general (kcal)
ENERGY_KJ_ID = 1062
PROTEIN_ID = 1003
FAT_ID = 1004
CARB_IDS = [1005, 1050]  # carbohydrate by difference, then by summation

UNAVAILABLE_HINT = (
    "USDA FoodData Central is unavailable right now. Tell the user the live lookup failed "
    "and to try again shortly. Do not make up nutrition values."
)


class USDAError(Exception):
    """A failure the model can act on.

    `unavailable` is True when USDA itself could not be used (no key, timeout, outage),
    as opposed to USDA answering "no such food". Callers use it to decide on a fallback.
    """

    def __init__(self, error: str, hint: str, unavailable: bool = False):
        super().__init__(error)
        self.error = error
        self.hint = hint
        self.unavailable = unavailable

    def to_dict(self) -> dict:
        return {"error": self.error, "hint": self.hint}


def _request(method: str, path: str, json_body: dict | None = None) -> dict | None:
    """Call USDA and return the parsed JSON, or None for a 404. Raises USDAError otherwise."""
    api_key = os.environ.get("USDA_API_KEY")
    if not api_key:
        raise USDAError("USDA_API_KEY is not set.", UNAVAILABLE_HINT, unavailable=True)

    # Exception messages from requests can contain the full URL, key included,
    # so they are never passed on.
    try:
        response = requests.request(
            method, BASE_URL + path, params={"api_key": api_key}, json=json_body, timeout=TIMEOUT_SECONDS
        )
    except requests.Timeout:
        raise USDAError(f"USDA did not answer within {TIMEOUT_SECONDS} seconds.", UNAVAILABLE_HINT, unavailable=True)
    except requests.RequestException:
        raise USDAError("Could not connect to USDA FoodData Central.", UNAVAILABLE_HINT, unavailable=True)

    if response.status_code == 404:
        return None
    if response.status_code == 403:
        raise USDAError("USDA rejected the API key.", UNAVAILABLE_HINT, unavailable=True)
    if response.status_code == 429:
        raise USDAError("USDA rate limit reached.", UNAVAILABLE_HINT, unavailable=True)
    if response.status_code != 200:
        raise USDAError(f"USDA returned HTTP {response.status_code}.", UNAVAILABLE_HINT, unavailable=True)

    try:
        return response.json()
    except ValueError:
        raise USDAError("USDA returned a response that is not JSON.", UNAVAILABLE_HINT, unavailable=True)


def _first_present(values: dict[int, float], nutrient_ids: list[int]) -> float | None:
    for nutrient_id in nutrient_ids:
        if values.get(nutrient_id) is not None:
            return values[nutrient_id]
    return None


def _per_100g(values: dict[int, float]) -> dict | None:
    """Turn {nutrient_id: amount per 100 g} into our four numbers.

    Returns None when energy, protein, carbs, or fat is missing: a gap is never filled with 0.
    """
    kcal = _first_present(values, ENERGY_KCAL_IDS)
    if kcal is None and values.get(ENERGY_KJ_ID) is not None:
        kcal = values[ENERGY_KJ_ID] / 4.184
    protein = values.get(PROTEIN_ID)
    fat = values.get(FAT_ID)
    carbs = _first_present(values, CARB_IDS)
    if None in (kcal, protein, fat, carbs):
        return None

    # "Carbohydrate by difference" can come out slightly negative for meats. Treat that as 0.
    return {
        "calories_kcal": round(kcal),
        "protein_g": round(max(protein, 0.0), 1),
        "carbs_g": round(max(carbs, 0.0), 1),
        "fat_g": round(max(fat, 0.0), 1),
    }


def _summarize(food: dict, values: dict[int, float]) -> dict:
    return {
        "fdc_id": food["fdcId"],
        "description": food["description"],
        "data_type": food.get("dataType"),
        "per_100g": _per_100g(values),
    }


def search_foods(query: str, data_types: list[str] = GENERIC_DATA_TYPES, page_size: int = 5) -> list[dict]:
    """Search USDA by name. Returns up to `page_size` foods in USDA's ranking order.

    A food's `per_100g` is None if USDA lacks one of the four numbers for it.
    """
    body = {"query": query, "dataType": data_types, "pageSize": page_size}
    data = _request("POST", "/foods/search", json_body=body) or {}

    # The search endpoint lists nutrients flat: {"nutrientId": 1008, "value": 130, ...}
    results = []
    for food in data.get("foods", []):
        values = {n["nutrientId"]: n.get("value") for n in food.get("foodNutrients", [])}
        results.append(_summarize(food, values))
    return results


def fetch_food(fdc_id: int) -> dict:
    """Fetch one food by FDC ID. Raises USDAError if the ID does not exist or has no usable data."""
    food = _request("GET", f"/food/{fdc_id}")
    if food is None:
        raise USDAError(
            f"USDA has no food with FDC ID {fdc_id}.",
            "Search by the food's common English name instead.",
        )

    # The detail endpoint nests nutrients: {"nutrient": {"id": 1008, ...}, "amount": 130}
    values = {n["nutrient"]["id"]: n.get("amount") for n in food.get("foodNutrients", []) if "nutrient" in n}
    result = _summarize(food, values)
    if result["per_100g"] is None:
        raise USDAError(
            f"USDA entry {fdc_id} ({food['description']}) is missing calories or macros.",
            "Search by the food's common English name instead.",
        )
    return result
