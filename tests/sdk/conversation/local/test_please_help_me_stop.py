from collections.abc import Sequence
from typing import ClassVar

from openhands.sdk import Action, Observation, TextContent
from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.llm import ImageContent, Message, MessageToolCall
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import Tool, ToolDefinition, ToolExecutor, register_tool


class _HelpAction(Action):
    message: str


class _HelpObservation(Observation):
    message: str = "Pending user's decision"

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class _HelpExecutor(ToolExecutor[_HelpAction, _HelpObservation]):
    def __call__(self, action: _HelpAction, conversation=None) -> _HelpObservation:
        return _HelpObservation()


class _HelpTool(ToolDefinition[_HelpAction, _HelpObservation]):
    name: ClassVar[str] = "please_help_me"

    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["_HelpTool"]:
        return [
            cls(
                description="Ask the human user for help.",
                action_type=_HelpAction,
                observation_type=_HelpObservation,
                executor=_HelpExecutor(),
            )
        ]


def _make_tool(conv_state=None, **kwargs) -> Sequence[ToolDefinition]:
    return _HelpTool.create(conv_state, **kwargs)


register_tool("please_help_me", _make_tool)


def test_please_help_me_stops_the_current_run() -> None:
    llm = TestLLM.from_messages(
        [
            Message(
                role="assistant",
                content=[TextContent(text="I need the human to solve the CAPTCHA.")],
                tool_calls=[
                    MessageToolCall(
                        id="call_help_1",
                        name="please_help_me",
                        arguments=(
                            '{"message": "Please solve the CAPTCHA, then reply ok."}'
                        ),
                        origin="completion",
                    )
                ],
            )
        ]
    )
    agent = Agent(llm=llm, tools=[Tool(name="please_help_me")])
    conversation = Conversation(agent=agent)

    conversation.send_message(
        Message(role="user", content=[TextContent(text="Continue the login flow")])
    )
    conversation.run()

    assert conversation.state.execution_status == ConversationExecutionStatus.FINISHED

    action_events = [
        event
        for event in conversation.state.events
        if isinstance(event, ActionEvent) and event.tool_name == "please_help_me"
    ]
    observation_events = [
        event
        for event in conversation.state.events
        if isinstance(event, ObservationEvent) and event.tool_name == "please_help_me"
    ]

    assert len(action_events) == 1
    assert len(observation_events) == 1
