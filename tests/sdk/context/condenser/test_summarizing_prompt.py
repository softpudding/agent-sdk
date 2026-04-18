"""Tests for the LLMSummarizingCondenser prompt template.

The template is shared across all use cases of the condenser, but the
OpenBrowser fork tunes it for browser/routine-replay workloads where the
dominant useful signal is "which SOP steps are done, which are pending,
and what is the current page state" — not the coding-task framing
(CODE_STATE / TESTS / CHANGES / DEPS) that the upstream template leads
with.
"""

from pathlib import Path

from openhands.sdk.context.prompts.prompt import render_template


PROMPT_DIR = (
    Path(__file__).resolve().parents[4]
    / "openhands-sdk"
    / "openhands"
    / "sdk"
    / "context"
    / "condenser"
    / "prompts"
)


def _render(events: list[str] | None = None) -> str:
    return render_template(
        prompt_dir=str(PROMPT_DIR),
        template_name="summarizing_prompt.j2",
        events=events or ["<event 1>", "<event 2>"],
    )


def test_summarizing_prompt_leads_with_browser_replay_framing() -> None:
    """The top-level tracking fields must emphasize browser-replay state —
    what has been done and what has not — instead of code-task fields."""
    message = _render()

    # Browser-replay / SOP-oriented fields lead the template.
    assert "USER_GOAL:" in message
    assert "STEPS_COMPLETED:" in message
    assert "STEPS_REMAINING:" in message
    assert "CURRENT_PAGE_STATE:" in message
    assert "OPEN_QUESTIONS:" in message


def test_summarizing_prompt_keeps_code_fields_as_optional_appendix() -> None:
    """Code-specific fields remain available for coding tasks but are
    clearly marked optional so they do not dominate browser replays."""
    message = _render()

    assert "For code-specific tasks, also include:" in message
    assert "CODE_STATE:" in message


def test_summarizing_prompt_has_browser_replay_example() -> None:
    """At least one worked example must frame the summary in
    browser-replay terms so small models pattern-match to the right
    shape."""
    message = _render()

    # A browser-replay example, not the haiku example.
    assert "For browser routine replay:" in message
    # The example must mention steps and page state explicitly.
    assert "STEPS_COMPLETED:" in message
    assert "STEPS_REMAINING:" in message


def test_summarizing_prompt_renders_events() -> None:
    """The events block must still render the list the condenser passes in."""
    message = _render(events=["alpha-event", "beta-event"])

    assert "<EVENT>" in message
    assert "alpha-event" in message
    assert "beta-event" in message
