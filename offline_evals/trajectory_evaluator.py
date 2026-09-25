import json
import os
import sys
from pathlib import Path

# Before the earliest future date in dataset.json (2026-11-01)
# and after the past-date example (2025-10-10).
os.environ["EVAL_TODAY"] = "2026-09-25"
# This eval scores tool calls only, so skip the live flight API.
os.environ["STUB_FLIGHTS"] = "1"

from agentevals.trajectory import create_trajectory_match_evaluator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import agent

load_dotenv()

DATASET_NAME = "flight-evals"

trajectory_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="unordered",
    tool_args_match_mode="exact",
)


def invoke_agent(inputs: dict) -> dict:
    result = agent.invoke({"messages": [HumanMessage(content=inputs["query"])]})
    return {"messages": result["messages"]}


def trajectory_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    return trajectory_evaluator(
        outputs=outputs["messages"],
        reference_outputs=reference_outputs["expected_trajectory"],
    )

def ensure_dataset(client: Client) -> None:
    examples = json.loads(Path(__file__).with_name("dataset.json").read_text())
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(DATASET_NAME, description="Flight agent evals")
    else:
        existing_ids = [e.id for e in client.list_examples(dataset_name=DATASET_NAME)]
        if existing_ids:
            client.delete_examples(existing_ids)
    client.create_examples(dataset_name=DATASET_NAME, examples=examples)


if __name__ == "__main__":
    client = Client()
    ensure_dataset(client)
    results = client.evaluate(
        invoke_agent,
        data=DATASET_NAME,
        evaluators=[trajectory_accuracy],
        experiment_prefix="flight-trajectory-eval-v1",
        max_concurrency=2,
        metadata={
            "agent": "openai/gpt-oss-20b",
            "change": "stubbed flight tool; pinned eval date",
        },
    )
    print(results.url)
