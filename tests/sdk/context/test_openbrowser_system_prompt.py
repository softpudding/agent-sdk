from pathlib import Path

from openhands.sdk.context.prompts.prompt import render_template


PROMPT_DIR = (
    Path(__file__).resolve().parents[3]
    / "openhands-sdk"
    / "openhands"
    / "sdk"
    / "agent"
    / "prompts"
)


def _render_system_prompt(**kwargs) -> str:
    return render_template(
        prompt_dir=str(PROMPT_DIR),
        template_name="system_prompt.j2",
        security_policy_filename="security_policy.j2",
        **kwargs,
    )


def test_openbrowser_system_prompt_lists_swipe_and_select_actions() -> None:
    message = _render_system_prompt()

    assert "`click` and `keyboard_input` require confirmation" in message
    assert "`hover`, `scroll`, `swipe`, and `select` execute directly" in message


def test_openbrowser_system_prompt_does_not_force_scroll_first() -> None:
    message = _render_system_prompt()

    assert 'do not declare it "not found" too early' in message
    assert (
        "Scroll is one exploration tool, not the default answer to every miss"
        in message
    )
    assert "ALWAYS try to SCROLL first" not in message
    assert "icon button next to a visible count or badge" in message
    assert "keep the same highlight mode and try the next page" in message
    assert (
        "Do not use icon-only tokens, guessed semantics from appearance, "
        "or inferred labels." in message
    )
    assert "If the target is not on highlight page 1, continue paginating" in message
    assert "Do not jump from a first-page miss to `keywords`" in message
    assert "After any significant page-state change" in message
    assert (
        "Do not jump away from `any` on that changed page before "
        "rebuilding the mixed-type inventory" in message
    )


def test_openbrowser_system_prompt_prefers_narrowing_over_first_match() -> None:
    message = _render_system_prompt()

    assert (
        "If multiple candidates still fit, narrow the search rather than guessing"
        in message
    )
    assert "Pick the first matching element" not in message
    assert (
        "Use keywords only when exact literal text you can already see on "
        "the current page can disambiguate the target" in message
    )


def test_openbrowser_system_prompt_makes_interaction_selection_deterministic() -> None:
    message = _render_system_prompt()

    assert "Keep interaction selection deterministic and evidence-based" in message
    assert (
        "High-level browsing can be open-ended, but target selection cannot be casual"
        in message
    )
    assert "You may use more flexible strategies" not in message
    assert (
        "When casually browsing: You can be more relaxed in your approach"
        not in message
    )


def test_openbrowser_system_prompt_limits_keywords_to_exact_observed_text() -> None:
    message = _render_system_prompt()

    assert (
        "Use `keywords` only when you can already see exact literal text "
        "characters on the current page" in message
    )
    assert (
        "If a control has visible readable text, you may use that exact text "
        "with `keywords`" in message
    )
    assert (
        "Never probe the page with guessed words like `next`, `previous`, "
        "`close`, `search`, `settings`, `gear`, or `bell`" in message
    )
    assert (
        "Use keywords for concrete text you can already see or reliably know exists"
        not in message
    )


def test_openbrowser_system_prompt_limits_agents_md_writes() -> None:
    message = _render_system_prompt()

    assert "Only add to `AGENTS.md` when the user asks for it" in message
    assert "Add important insights, patterns, and learnings to this file" not in message


def test_openbrowser_system_prompt_softens_troubleshooting_dump() -> None:
    message = _render_system_prompt()

    assert "reflect on a few plausible sources of the problem" in message
    assert "without turning the response into a long diagnostic dump" in message
    assert "5-7 different possible sources of the problem" not in message


def test_openbrowser_system_prompt_describes_snapshot_scoped_highlight_ids() -> None:
    message = _render_system_prompt()

    assert (
        "Every `highlight` response returns a `highlight_snapshot_id` plus "
        'page-local element IDs such as "1", "2", "3"' in message
    )
    assert (
        "Element IDs are only valid together with the exact "
        "`highlight_snapshot_id` that produced them" in message
    )
    assert (
        "To continue pagination on the same unchanged page state, pass the "
        "previous `highlight_snapshot_id` back into the next `highlight` call"
        in message
    )
    assert (
        "The ORANGE confirmation preview does not create a new `highlight_snapshot_id`"
        in message
    )
    assert 'Element IDs (e.g., "a3f2b1") are your interface to the page' not in message


def test_openbrowser_system_prompt_explains_why_any_is_first() -> None:
    message = _render_system_prompt()

    assert (
        '`highlight` with `element_type: "any"` is the default first pass for '
        "each new page state" in message
    )
    assert "extension-derived page insight across element types" in message
    assert "authoritative first-pass inventory for each new page state" in message
    assert (
        "carries extension-derived structure and cross-type context that "
        "narrower passes can hide" in message
    )
