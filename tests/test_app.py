"""Tests for the harness in app.py. The model is replaced by scripted replies, so nothing is called."""

import json
from types import SimpleNamespace

import app


class FakeReply:
    """Stands in for the message object LiteLLM returns."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


def tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_run_agent_never_records_a_garbled_tool_name(monkeypatch):
    args = {"food_name": "congee", "quantity": 1, "unit": "bowl"}
    replies = [
        FakeReply(tool_calls=[
            tool_call("1", "لمات_portion_size", args),  # garbled: not a real tool
            tool_call("2", "x:default_api:estimate_portion_size", args),  # junk prefix: cleaned
        ]),
        FakeReply(content="Estimated: ..."),
    ]
    monkeypatch.setattr(app.litellm, "completion", lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))]))

    response, tool_calls = app.run_agent([{"role": "user", "content": "half a bowl of congee"}])

    assert response == "Estimated: ..."
    assert [c["name"] for c in tool_calls] == ["unknown_tool", "estimate_portion_size"]
    assert "error" in json.loads(tool_calls[0]["result"])  # the error result is kept
    assert json.loads(tool_calls[1]["result"])["food_name"] == "congee"


def test_run_agent_answers_a_tool_call_whose_arguments_are_not_json(monkeypatch):
    bad_call = SimpleNamespace(id="1", function=SimpleNamespace(name="estimate_portion_size", arguments="{food_name: congee"))
    replies = [
        FakeReply(tool_calls=[bad_call]),
        FakeReply(content="Estimated: ..."),
    ]
    monkeypatch.setattr(app.litellm, "completion", lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))]))
    messages = [{"role": "user", "content": "half a bowl of congee"}]

    response, tool_calls = app.run_agent(messages)

    assert response == "Estimated: ..."
    assert tool_calls[0]["name"] == "estimate_portion_size"
    assert tool_calls[0]["args"] == {}
    assert "error" in json.loads(tool_calls[0]["result"])
    assert any(m["role"] == "tool" and m["tool_call_id"] == "1" for m in messages)
