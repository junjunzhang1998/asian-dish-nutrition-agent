# East Asian Food Nutrition Agent

**Live:** [Cloud Run URL](https://asian-dish-nutrition-agent-git-410194130748.northamerica-northeast1.run.app/) (sign in with a Columbia account)

I recently started tracking my diet and noticed that calorie-tracking apps on the market often struggle with Asian food. The three main problems I found were:
- **Limited data:** it is easy to find information for common American dishes, but not for specific Asian dishes.
- **Inaccurate entries:** the same dish can vary a lot depending on the ingredients, recipe, and cooking method.
- **Portion sizes are hard to estimate:** in Asian culture, many meals are shared family-style. I don't remember the exact weight of everything I ate, but I can describe it in everyday language, such as 6 dumplings, a bowl of rice, or a plate of stir-fried greens.

So I created this agent, which lets users describe what they ate in everyday language and returns a calorie and macro estimate. I focused on East Asian food because these are the cuisines I eat day-to-day, and their nutrition is hard to track given the problems above. Because recipes and portion sizes are uncertain, the agent gives a low/typical/high range instead of a single number, and asks a follow-up question to narrow it down.

## Tools

I created three tools for the agent:
- **`lookup_food_nutrition`** — Looks up calories and macros per 100 g for a single food or ingredient. This tool calls the USDA FoodData Central API live. It is used when the user asks about or compares specific ingredients (e.g. tofu vs. salmon).
- **`estimate_portion_size`** — Converts portions in everyday language like "6 dumplings" or "half a bowl of rice" into a low / typical / high weight range in grams. It is used when the user describes an amount without a weight. This is an original tool.
- **`estimate_dish_nutrition`** — Estimates calories and macros for a cooked dish like mapo tofu. It takes the portion size, the recipe ingredients, and how oily the dish is, and gives back a low / typical / high range. The nutrition values for the ingredients come from USDA. The agent uses this for dishes, not single ingredients. This is an original tool.


## How it works

When you describe a dish, the agent first converts the amount you gave (a plate, a bowl, 6 pieces) into grams using `estimate_portion_size` tool. Then `estimate_dish_nutrition` takes a built-in recipe for that dish and scales it to that weight. If I haven't written a recipe for the dish (15 common Asian dish recipes in .json format), the model comes up with an ingredient list and the tool uses that. The nutrition values for each ingredient come from USDA.

The answer is a low / typical / high calorie range, since I usually don't know exactly how big the portion was or how much oil went in. Macros are based on the typical estimate. After each estimate, the agent asks one question about size or oil to narrow the range.

If you only ask about one ingredient, like the protein in tofu, it goes straight to `lookup_food_nutrition`.


## Sample queries

1. **"I'm deciding between tofu and salmon for dinner. Which one is higher in protein, and how do their calories and macros compare?"**  
   *Expected:* `lookup_food_nutrition` runs once per food, and the reply compares them per 100 g using USDA data.

2. **"I'm trying to log my lunch, but I didn't weigh anything. I had 6 dumplings and about half a bowl of rice. How many calories did I eat?"**  
   *Expected:* `estimate_portion_size` converts the everyday portions into gram ranges, then `estimate_dish_nutrition` gives a low/typical/high calorie range. The agent asks one follow-up question; answering it narrows the range.

3. **"I weighed my leftover mapo tofu and it was about 300g. It was homemade and fairly oily. Roughly how many calories and macros should I log?"** Then: **"Also add a bowl of rice."**  
   *Expected:* `estimate_dish_nutrition` uses the known weight, the recipe ingredients and the oil level. After the second message, the agent remembers the mapo tofu and gives a combined total.


## Known limits

- Built-in recipes represent one version of a dish; restaurant and homemade recipes vary.
- Dishes without a built-in recipe rely on an estimated ingredient list and are lower confidence.
- USDA searches may sometimes use a close match rather than an exact food entry.
