"""Tests for qwen3-coder-style XML tool-call recovery.

Some small qwen variants (notably qwen3.5-flash on DashScope) fall back to
emitting tool calls in the qwen3-coder training format as plain text inside
`reasoning_content` rather than as structured `tool_calls`. The recovery
helper converts these back into canonical `MessageToolCall`s and strips the
XML from the text fields, so the conversation history is well-formed.
"""

import json

from openhands.sdk.llm.message import (
    Message,
    TextContent,
    recover_qwen_xml_tool_calls,
)


# Real reasoning_content from a broken session
# (~/.openbrowser/conversations/f98378c1.../events/event-00066-*.json),
# qwen3.5-flash on DashScope.
REAL_EVENT_66_RC = (
    "<tool_call>\n"
    "<function=mouse>\n"
    "<parameter=action>\n"
    "click\n"
    "</parameter>\n"
    "<parameter=coordinate>\n"
    "[560, 294]\n"
    "</parameter>\n"
    "<parameter=security_risk>\n"
    "LOW\n"
    "</parameter>\n"
    "<parameter=summary>\n"
    "Click June 17, 2026 as check-out date in calendar\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _msg(rc=None, content_text=""):
    content = [TextContent(text=content_text)] if content_text else []
    return Message(role="assistant", content=content, reasoning_content=rc)


def test_recovers_real_qwen_flash_session_event_66():
    msg = _msg(rc=REAL_EVENT_66_RC)
    out = recover_qwen_xml_tool_calls(msg)

    assert out.tool_calls is not None and len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "mouse"
    assert tc.origin == "completion"
    args = json.loads(tc.arguments)
    assert args["action"] == "click"
    assert args["coordinate"] == [560, 294]  # JSON-array param decoded
    assert args["security_risk"] == "LOW"
    assert "Jun" in args["summary"]
    # XML stripped from reasoning_content
    assert out.reasoning_content in (None, "")


def test_preserves_surrounding_prose():
    rc = (
        "Looking at the screenshot, I should click June 17.\n"
        + REAL_EVENT_66_RC
        + "\nNow I'll wait for the popup."
    )
    out = recover_qwen_xml_tool_calls(_msg(rc=rc))

    assert out.tool_calls and len(out.tool_calls) == 1
    assert out.reasoning_content
    assert "<tool_call>" not in out.reasoning_content
    assert "click June 17" in out.reasoning_content
    assert "wait for the popup" in out.reasoning_content


def test_multiple_tool_calls_in_one_message():
    rc = REAL_EVENT_66_RC + "\n" + REAL_EVENT_66_RC.replace("click", "move")
    out = recover_qwen_xml_tool_calls(_msg(rc=rc))

    assert out.tool_calls and len(out.tool_calls) == 2
    actions = [json.loads(tc.arguments)["action"] for tc in out.tool_calls]
    assert actions == ["click", "move"]


def test_recovers_from_content_text_when_reasoning_empty():
    out = recover_qwen_xml_tool_calls(_msg(content_text=REAL_EVENT_66_RC))

    assert out.tool_calls and len(out.tool_calls) == 1
    # XML-only content gets emptied; non-XML content survives
    text_parts = [c.text for c in out.content if isinstance(c, TextContent)]
    assert all("<tool_call>" not in t for t in text_parts)


def test_hallucinated_tool_name_still_parses():
    """qwen-flash sometimes calls a non-existent `think` tool. The parser
    should still produce a structured call so the dispatcher can return a
    real error rather than the agent loop spinning on empty messages."""
    rc = (
        "<tool_call>\n"
        "<function=think>\n"
        "<parameter=thought>\nplanning next step\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = recover_qwen_xml_tool_calls(_msg(rc=rc))
    assert out.tool_calls and out.tool_calls[0].name == "think"


def test_no_xml_passthrough():
    out = recover_qwen_xml_tool_calls(_msg(rc="just thinking, no tool call"))
    assert out.tool_calls is None
    assert out.reasoning_content == "just thinking, no tool call"


def test_existing_tool_calls_not_overwritten():
    """If structured tool_calls are already present (the normal path), the
    recovery helper must not touch the message even if XML happens to be
    in reasoning_content."""
    from openhands.sdk.llm.message import MessageToolCall

    msg = Message(
        role="assistant",
        content=[],
        reasoning_content=REAL_EVENT_66_RC,
        tool_calls=[
            MessageToolCall(
                id="real_id",
                name="keyboard",
                arguments=json.dumps({"text": "hi"}),
                origin="completion",
            )
        ],
    )
    out = recover_qwen_xml_tool_calls(msg)
    assert out is msg  # unchanged reference
    assert out.tool_calls and out.tool_calls[0].name == "keyboard"


def test_malformed_xml_passthrough():
    """Unclosed <tool_call> block: parser silently skips, text preserved."""
    rc = "<tool_call>\n<function=mouse>\n<parameter=action>click\n"
    out = recover_qwen_xml_tool_calls(_msg(rc=rc))
    assert out.tool_calls is None
    assert out.reasoning_content == rc
