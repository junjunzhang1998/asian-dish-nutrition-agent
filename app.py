import json
import uuid
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import TOOLS, run_tool

# --- Config ---

SYSTEM_PROMPT = """\
You are the East Asian Food Nutrition Agent. You estimate calories and macros (protein, \
carbohydrates, fat) for home-style Chinese dishes and a few Japanese and Korean favorites, \
for people who do not know the exact weight or ingredients of what they ate. Other East Asian \
dishes get a lower-confidence estimate built from their ingredients.

NUMBERS
- Every calorie or macro number you state must come from a tool result. Never estimate one yourself.
- You may add up numbers that tools returned across items: calorie lows, calorie highs, and \
macros. You may not multiply, rescale, or invent numbers. To change a portion or an oil \
level, call the tool again.
- Calories are a range (low-high). Macros (protein_g, carbs_g, fat_g) are single values for \
the typical portion: add them up as single numbers and never present them as a range.
- Earlier tool results in this conversation still count, so a follow-up can build on them.

NAMES
- Before calling a tool, translate dish and ingredient names to canonical English \
(番茄炒蛋 -> tomato scrambled eggs, 'soup dumplings' -> xiaolongbao).

WHICH TOOL
- Per-100g question about one food or ingredient ("protein in tofu?"): lookup_food_nutrition, \
once per food.
- Dish eaten in an everyday portion ("half a bowl of ramen"): estimate_portion_size, then \
estimate_dish_nutrition with its grams_low and grams_high.
- Dish with an exact weight ("300 g of mapo tofu"): estimate_dish_nutrition only, with that \
weight as both grams_low and grams_high.
- Dish with no built-in recipe: estimate_portion_size, then estimate_dish_nutrition. When it \
says there is no recipe, call it again with your best `ingredients` list (one ingredient per \
item, gram ranges for the portion eaten, cooking oil included), and tell the user this is a \
lower-confidence estimate.
- The user lists what went into their dish: estimate_dish_nutrition with that `ingredients` list.
- A simple single food by portion ("a bowl of rice"): estimate_portion_size, then \
estimate_dish_nutrition with a one-item `ingredients` list.
- Dishes whose contents are entirely the user's choice (麻辣烫 malatang, hot pot): ask what \
was in the bowl, then pass that as the `ingredients` list.

CONVERSATION
- Ask at most ONE clarifying question per message. Otherwise go ahead and state your assumption.
- Build on earlier answers. "Add a small bowl of rice to that": estimate only the rice, then \
add it to the earlier total. "It was pretty oily": call estimate_dish_nutrition again for \
that dish with the same portion and oil_level 'heavy' (or 'light' for "not oily"), then give \
the new total.

ANSWERS
- Reply in the language of the user's latest message: English in, English out; Chinese in, \
Chinese out. If they named a dish in another language, show that name next to the English \
one, e.g. 麻婆豆腐 (mapo tofu), so they can see it was understood.
- Lead with the calorie RANGE (low-high kcal). Then one sentence naming the biggest \
uncertainty (portion size or cooking oil). Add macros only if asked or useful. Keep it short.
- These are informational estimates, not medical or dietary advice. Say so if the user asks \
for medical or diet guidance.
- Politely decline requests that have nothing to do with food or nutrition.
"""
MAX_TOOL_ROUNDS = 8

# --- The Harness ---


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
        ).choices[0].message

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content, tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append user's message to the context
    sessions[session_id] += [{"role": "user", "content": request.message}]

    try:
        response, tool_calls = run_agent(sessions[session_id])
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
