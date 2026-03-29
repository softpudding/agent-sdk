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


def test_openbrowser_system_prompt_describes_default_interactive_observations() -> None:
    message = _render_system_prompt()

    assert (
        "Outside of the YELLOW confirmation preview for `click` and "
        "`keyboard_input`, when a browser action returns a screenshot and that "
        "action is not `tab view`, treat that screenshot as the default "
        '`highlight` `element_type: "any"` page 1 observation' in message
    )
    assert (
        "Treat the returned `element_id`s as the working inventory of "
        "currently visible interactive elements" in message
    )
    assert (
        "Use `tab view` only when you explicitly need the clean raw screenshot "
        "without overlays" in message
    )
    assert (
        "Use `highlight` deliberately for pagination, type-specific discovery, "
        "or rebuilding the mixed-type inventory after the page state changed" in message
    )
    assert "highlight_snapshot_id" not in message


def test_openbrowser_system_prompt_uses_yellow_confirmation_language() -> None:
    message = _render_system_prompt()

    assert "### Stage 2: YELLOW - Action Confirmation" in message
    assert (
        "The YELLOW confirmation preview keeps the chosen `element_id` pending "
        "for confirmation" in message
    )
    assert "ORANGE" not in message


def test_openbrowser_system_prompt_requires_element_id_grounded_reasoning() -> None:
    message = _render_system_prompt()

    assert (
        "In your reasoning, explicitly name the candidate `element_id` values "
        "you are considering before you act" in message
    )
    assert (
        'Do not reason only in vague spatial language like "the button on the '
        'right" when a visible `element_id` exists' in message
    )
    assert (
        'Good: "`A1H` is the likely Search button because it is in the '
        "top-right toolbar" in message
    )
    assert (
        "No matching `element_id` appears in the current `any` page 1 "
        "observation" in message
    )
    assert (
        'Bad: "I will click the top-right button" without naming the visible '
        "`element_id`." in message
    )


def test_openbrowser_system_prompt_encourages_detail_views_for_preview_items() -> None:
    message = _render_system_prompt()

    assert (
        "Feeds, search results, grids, cards, and post previews are often "
        "staging views rather than the best place to finish a task." in message
    )
    assert (
        "Opening the card/post/detail view is encouraged when it gives you "
        "clearer targets, fuller context, or more reliable controls." in message
    )
    assert (
        "actions such as like, favorite, comment, follow, share, or open are "
        "usually still available there with better context" in message
    )
    assert (
        "This card only shows a preview, so opening the post detail is the "
        "flexible move" in message
    )


def test_openbrowser_system_prompt_repositions_occluded_targets_with_scroll() -> None:
    message = _render_system_prompt()

    assert (
        "If the intended control is visible but cramped, near a viewport edge, "
        "or partially occluded by sticky UI, floating bars, badges, or "
        "overlays, scroll to reposition it before acting." in message
    )
    assert (
        "Scroll to improve geometry when the target is partly occluded, "
        "squeezed against another element, or too close to a sticky "
        "header/footer or viewport edge" in message
    )
    assert (
        "Scroll is also for geometry, not just discovery: use it to move a "
        "partly covered target into a cleaner, more clickable position" in message
    )
    assert (
        "The target button is visible but crowded by neighboring UI, so I "
        "should scroll a bit to reposition it before clicking." in message
    )


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


def test_openbrowser_system_prompt_uses_help_tool_for_captcha() -> None:
    message = _render_system_prompt()

    assert "call `please_help_me`" in message
    assert "do NOT only say it in assistant text" in message
    assert "wait for the user's next message" in message
