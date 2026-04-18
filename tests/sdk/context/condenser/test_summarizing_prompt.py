"""Tests for the LLMSummarizingCondenser prompt template.

The template must work for any interactive-agent task — browser routine
replays, free-form browser sessions, and coding tasks — so the top-level
framing is intentionally generic (goal / progress / remaining /
current state / blockers). Task-specific framing (SOP steps, CODE_STATE,
etc.) is opt-in.
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


def test_summarizing_prompt_leads_with_generic_progress_framing() -> None:
    """The top-level tracking fields must be generic enough for any
    task type — replay, free-form browser work, or coding.
    """
    message = _render()

    assert "USER_GOAL:" in message
    assert "COMPLETED:" in message
    assert "PENDING:" in message
    assert "CURRENT_STATE:" in message
    assert "OPEN_QUESTIONS:" in message
    assert "TASK_TRACKING:" in message


def test_summarizing_prompt_is_not_routine_replay_specific() -> None:
    """The top-level instructions must not assume the task is a routine
    replay. Compaction runs in free-form sessions too, and wording that
    mentions SOPs or Step sections as the default case misleads the
    summarizer on non-replay tasks.
    """
    message = _render()

    # The top-level framing must not claim the task is (or most tasks
    # are) a routine replay or SOP.
    assert "Most of these tasks are Browser Routine replays" not in message
    assert "the user sent a Markdown SOP" not in message
    assert "numbered `## Step N:` sections" not in message


def test_summarizing_prompt_keeps_code_fields_as_optional_appendix() -> None:
    """Code-specific fields remain available for coding tasks but are
    clearly marked optional."""
    message = _render()

    assert "For code-specific tasks, also include:" in message
    assert "CODE_STATE:" in message


def test_summarizing_prompt_examples_are_not_task_specific() -> None:
    """The examples must not hard-code the content of any specific user
    task (e.g. the Finviz routine used for debugging). Generic placeholders
    keep the prompt from over-fitting to one benchmark scenario.
    """
    message = _render()

    # Specific routine / site content the debug session used must not
    # leak into the production prompt.
    assert "value-stocks-monthly-drop" not in message
    assert "Finviz" not in message
    assert "finviz" not in message
    assert "fs_cap" not in message
    assert "fs_fa_pb" not in message
    assert "smallover" not in message


def test_summarizing_prompt_renders_events() -> None:
    """The events block must still render the list the condenser passes in."""
    message = _render(events=["alpha-event", "beta-event"])

    assert "<EVENT>" in message
    assert "alpha-event" in message
    assert "beta-event" in message
