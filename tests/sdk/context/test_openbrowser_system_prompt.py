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
    assert (
        "older screenshots may later be dropped from live context while the "
        "reasoning text remains." in message
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
    assert "Follow these browser rules strictly:" not in message


def test_openbrowser_system_prompt_omits_routine_replay_block_by_default() -> None:
    """Outside routine-replay mode the block must be absent for both model
    sizes, so free-form conversations never see the keyword-unlock rules.
    """
    for kwargs in ({}, {"small_model": True}):
        message = _render_system_prompt(**kwargs)
        assert "<ROUTINE_REPLAY>" not in message
        assert "Keywords:" not in message


def test_openbrowser_system_prompt_routine_replay_block_is_present_when_enabled() -> (
    None
):
    """When routine_replay_mode is True, the block renders with fixed rules
    for using a Routine step's `**Keywords:**` token. The block must not
    depend on the model judging anything about message content.
    """
    for kwargs in ({}, {"small_model": True}):
        message = _render_system_prompt(routine_replay_mode=True, **kwargs)

        assert "<ROUTINE_REPLAY>" in message
        assert "This conversation is a routine replay." in message
        # The "trust the Routine" framing must lead the block. The Routine
        # is a recorded successful execution; following it literally is
        # more reliable than re-deriving targets visually at runtime.
        assert "Trust the Routine." in message
        assert (
            "It is a recorded successful execution that the compiler agent "
            "already analyzed against the original trace" in message
        )
        assert (
            "Whenever the Routine and your own reasoning about the current "
            "observation disagree about how to proceed, the Routine wins." in message
        )
        # The step-level Keywords hint with its fixed rules.
        assert "**Keywords:** <token>" in message
        # The trigger is step entry, not "first highlight call". Anchoring
        # on highlight let the model rationalize away the rule whenever it
        # decided not to call highlight for a step at all (session
        # 077abd52: clicked W5N from carry-over inventory and never reached
        # the rule).
        assert (
            "your VERY FIRST tool call for that step MUST be `highlight` "
            "with that token passed verbatim as the `keywords` argument" in message
        )
        # The override must name BOTH the large-model and small-model
        # rules that compete with "trust the routine". Naming only WORKFLOW
        # left an open back door for the small-model "if the current
        # observation already contains a clean matching `element_id`, use
        # it" rule, which is the small-model twin of the same trap.
        assert (
            'The general "act on the current observation instead of '
            'reflexively calling `highlight`" guidance from `<WORKFLOW>`' in message
        )
        assert (
            '"current observation before new discovery" priority from '
            "`<LARGE_MODEL_GUIDANCE>`" in message
        )
        assert (
            '"if the current observation already contains a clean matching '
            '`element_id`, use it" rule from `<SMALL_MODEL_GUIDANCE>`' in message
        )
        assert (
            "are ALL superseded on entry to a step that has a "
            "`**Keywords:**` line" in message
        )
        assert "The Routine token outranks the carry-over inventory." in message
        # The token is the primary anchor — not the visible label, not the
        # step title. Session 077abd52 also showed the model substituting
        # `["P/E"]` for `["fs_fa_pe"]` on a forced retry.
        assert (
            "Do not substitute the visible label, the step title, or any "
            "other paraphrase as the `keywords` argument" in message
        )
        assert "Use the `Keywords:` token ONLY on the step it belongs to." in message
        assert "If the step has no `**Keywords:**` line, do not invent one" in message
        # Fallback behavior on zero matches.
        assert (
            "If a highlight call with the `Keywords:` token returns zero matches"
            in message
        )
        assert "call `please_help_me` rather than guessing" in message
        # The OLD self-contradicting carve-out paragraph (which stated the
        # visible-text rule and then negated it) must not be re-introduced.
        # Replay mode states one consistent rule by gating
        # DISCOVERY_STRATEGY at the source.
        assert "The general rule that `keywords` must be text visible" not in message
        # The no-guessing rule must still be enforced everywhere else.
        assert (
            "Never probe the page with guessed words like `next`, `previous`, "
            "`close`, `search`, `settings`, `gear`, or `bell`." in message
        )


def test_openbrowser_system_prompt_replay_keywords_rule_includes_token() -> None:
    """In routine_replay_mode the DISCOVERY_STRATEGY keyword rule must be the
    inclusive form that lists both verifiable sources (visible text and the
    Routine step's `**Keywords:**` token), instead of the strict visible-only
    form that the non-replay rendering uses."""
    message = _render_system_prompt(routine_replay_mode=True)

    assert (
        "Use `keywords` only with text you can verify verbatim: either the "
        "active routine step's `**Keywords:** <token>` value, or exact "
        "literal characters already visible on the current page." in message
    )
    # The strict non-replay phrasing must NOT appear — it would re-introduce
    # the contradiction the carve-out paragraph used to bridge.
    assert (
        "Use `keywords` only when you can already see exact literal text "
        "characters on the current page" not in message
    )


def test_openbrowser_system_prompt_non_replay_keywords_rule_stays_strict() -> None:
    """Outside replay mode the DISCOVERY_STRATEGY keyword rule must remain
    the strict visible-text-only form."""
    message = _render_system_prompt()

    assert (
        "Use `keywords` only when you can already see exact literal text "
        "characters on the current page" in message
    )
    # The replay-mode inclusive form must NOT leak into non-replay rendering.
    assert "active routine step's `**Keywords:** <token>` value" not in message


def test_openbrowser_system_prompt_uses_explicit_small_model_guidance() -> None:
    message = _render_system_prompt(small_model=True)

    assert "Follow these browser rules strictly:" in message
    assert "<SMALL_MODEL_GUIDANCE>" in message
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
