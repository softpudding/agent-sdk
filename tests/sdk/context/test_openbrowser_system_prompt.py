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


def test_openbrowser_system_prompt_stays_browser_focused() -> None:
    message = _render_system_prompt()

    assert "browser automation expert" in message
    assert "Use browser tools for web tasks" in message
    assert "When committing changes, use `git status`" not in message
    assert "When terminating processes:" not in message
    assert "If you encounter missing dependencies:" not in message


def test_openbrowser_system_prompt_lists_browser_tools() -> None:
    message = _render_system_prompt()

    assert "- `tab`: manage browser tabs" in message
    assert "- `highlight`: discover visible interactive elements" in message
    assert "- `element_interaction`: interact with visible elements" in message
    assert "- `dialog`: handle browser dialogs" in message


def test_openbrowser_system_prompt_describes_visual_grounding() -> None:
    message = _render_system_prompt()

    assert (
        "Outside of the YELLOW confirmation preview for `click` and "
        "`keyboard_input`, when a browser action returns a screenshot" in message
    )
    assert (
        "treat that screenshot as the default `highlight` "
        '`element_type: "any"` page 1 observation' in message
    )
    assert (
        "Treat the returned `element_id`s as the working inventory of "
        "currently visible interactive elements." in message
    )
    assert (
        "Use `tab view` only when you explicitly need the clean raw screenshot "
        "without overlays." in message
    )


def test_openbrowser_system_prompt_prefers_visual_discovery_before_keywords() -> None:
    message = _render_system_prompt()

    assert "Treat `any` as the default first pass for a new page state" in message
    assert (
        "If the target or a likely candidate is already partly visible, clipped "
        "by the viewport edge, or partly occluded, scroll to reposition it "
        "before continuing highlight pagination." in message
    )
    assert (
        "If the target is not on highlight page 1 and the page state is unchanged, "
        "continue paginating" in message
    )
    assert (
        "If dense UI, a sidebar, a tab strip, or collision-aware label placement "
        "may have split a likely target across highlight pages" in message
    )
    assert (
        "Use `keywords` only when you can already see exact literal text "
        "characters on the current page" in message
    )
    assert (
        "Never probe the page with guessed words like `next`, `previous`, "
        "`close`, `search`, `settings`, `gear`, or `bell`." in message
    )


def test_openbrowser_system_prompt_requires_element_id_grounded_reasoning() -> None:
    message = _render_system_prompt()

    assert (
        "In your reasoning, explicitly name the candidate `element_id` values "
        "you are considering before you act." in message
    )
    assert (
        'Do not reason only in vague spatial language like "the button on the '
        'right" when a visible `element_id` exists.' in message
    )
    assert (
        'Good examples:\n- "`A1H` is the likely Search button because it is in '
        "the top-right toolbar" in message
    )
    assert (
        'Bad example:\n- "I will click the top-right button" without naming the '
        "visible `element_id`." in message
    )


def test_openbrowser_system_prompt_uses_yellow_confirmation_language() -> None:
    message = _render_system_prompt()

    assert "### Stage 2: YELLOW - Action Confirmation" in message
    assert (
        "The YELLOW confirmation preview keeps the chosen `element_id` pending "
        "for confirmation" in message
    )
    assert (
        "verify visual position first and use any explicit preview text or "
        "metadata only as supporting evidence" in message
    )


def test_openbrowser_system_prompt_guides_detail_views_and_scroll() -> None:
    message = _render_system_prompt()

    assert 'Do not declare something "not found" too early.' in message
    assert (
        "If the target or a likely candidate is already visible but poorly "
        "positioned for interaction, prefer repositioning it over continuing "
        "discovery." in message
    )
    assert (
        "A partly visible target near the viewport edge should usually be "
        "repositioned closer to the middle of the viewport before clicking." in message
    )
    assert (
        "Treat a clipped or partly occluded target as a geometry problem first, "
        "not a discovery problem." in message
    )
    assert (
        "Opening the card/post/detail view is encouraged when it gives you "
        "clearer targets, fuller context, or more reliable controls." in message
    )
    assert (
        "Scroll to improve geometry when the target is visible but cramped, "
        "near a viewport edge, or partially occluded" in message
    )
    assert "Scroll is also for geometry, not just discovery" in message


def test_openbrowser_system_prompt_uses_help_tool_for_captcha() -> None:
    message = _render_system_prompt()

    assert "call `please_help_me`" in message
    assert "do NOT only say it in assistant text" in message
    assert "wait for the user's next message" in message


def test_openbrowser_system_prompt_uses_large_model_guidance_by_default() -> None:
    message = _render_system_prompt()

    assert "Use judgment, but keep these priorities:" in message
    assert "- current observation before new discovery" in message
    assert (
        "- geometry fixes before more discovery when the target is already "
        "partly visible" in message
    )
    assert "Use this browser SOP strictly:" not in message


def test_openbrowser_system_prompt_uses_explicit_small_model_sop() -> None:
    message = _render_system_prompt(small_model=True)

    assert "Use this browser SOP strictly:" in message
    assert (
        "If the target or a likely candidate is already partly visible, clipped "
        "by the viewport edge, or crowded by sticky UI, scroll first to "
        "reposition it closer to the middle of the viewport." in message
    )
    assert "If page 1 misses the target and the page state is unchanged" in message
    assert (
        "Only after pagination is insufficient, choose one of these next moves:"
        in message
    )
    assert (
        "If dense UI, a sidebar, a tab strip, or collision-aware label placement "
        "may have split the target across pages, keep paginating that same mode "
        "before changing strategy." in message
    )
    assert (
        "Prefer geometry fixes before more discovery when the target is already "
        "visible but poorly positioned." in message
    )
    assert (
        "use `keywords` only for exact visible text already on the current page"
        not in message
    )
    assert "Use judgment, but keep these priorities:" not in message
