import sys
from pathlib import Path

from agentevals.trajectory.match import create_trajectory_match_evaluator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import agent

load_dotenv()

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
)
# strict = Exact match of message structure and tool calls in the same order (message content can differ)
# unordered = Same message structure and tool calls as reference, but tool calls can happen in any order,
# subset = Agent calls only tools from reference (no extras),
# superset = Agent calls at least the reference tools (extras allowed),

def test_weather_tool_called_strict():
    result = agent.invoke({
        "messages": [HumanMessage(content="Suggest a flight from san francisco to tokyo for November 30th, 2026?")]
    })

    reference_trajectory = [
        HumanMessage(content="Suggest a flight from san francisco to tokyo for November 30th, 2026?"),
        # AIMessage(content="", tool_calls=[
        #     {"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}
        # ]),
        # ToolMessage(content="It's 75 degrees and sunny in San Francisco.", tool_call_id="call_1"),
        AIMessage(content="I found a flight from san francisco to tokyo for November 30th, 2026 for $1000."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )
    print(result["messages"])
    print("--------------------------------")
    print(reference_trajectory)
    print("--------------------------------")
    print(evaluation)
    assert evaluation["score"] is True

    print("Test passed")


if __name__ == "__main__":
    test_weather_tool_called_strict()