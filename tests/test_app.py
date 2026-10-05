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


def fake_model(monkeypatch, replies):
    """Replace the model with scripted replies. Returns the list of messages each call was sent."""
    sent = []

    def completion(**kwargs):
        sent.append(list(kwargs["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0), finish_reason="stop")])

    monkeypatch.setattr(app.litellm, "completion", completion)
    return sent


def test_run_agent_asks_again_after_an_empty_reply(monkeypatch, capsys):
    sent = fake_model(monkeypatch, [
        FakeReply(content=None),  # no text and no tool calls
        FakeReply(content="Estimated: ..."),
    ])
    messages = [{"role": "user", "content": "half a bowl of congee"}]

    response, tool_calls = app.run_agent(messages)

    assert response == "Estimated: ..."
    assert "finish_reason: stop" in capsys.readouterr().out
    assert app.EMPTY_REPLY_NUDGE not in sent[0]
    assert sent[1][-1] == app.EMPTY_REPLY_NUDGE  # the retry is told its last reply was empty
    assert app.EMPTY_REPLY_NUDGE not in messages  # but the session never keeps it
    assert not any(m["role"] == "assistant" and not (m["content"] or "").strip() for m in messages)


def test_run_agent_gives_up_after_three_empty_replies(monkeypatch):
    sent = fake_model(monkeypatch, [
        FakeReply(content=None), FakeReply(content=""), FakeReply(content="  "), FakeReply(content="never asked for"),
    ])
    messages = [{"role": "user", "content": "half a bowl of congee"}]

    response, tool_calls = app.run_agent(messages)

    assert response == app.EMPTY_REPLY_MESSAGE
    assert len(sent) == 3
    assert app.EMPTY_REPLY_NUDGE not in messages
