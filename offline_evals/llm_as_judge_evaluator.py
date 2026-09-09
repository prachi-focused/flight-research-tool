import json
import os
import sys
from pathlib import Path

from agentevals.trajectory.llm import (
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    create_trajectory_llm_as_judge,
)
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import agent

load_dotenv()

DATASET_NAME = "flight-evals"

model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

judge = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    judge=model,
)


def invoke_agent(inputs: dict) -> dict:
    result = agent.invoke({"messages": [HumanMessage(content=inputs["query"])]})
    return {"messages": result["messages"]}


def trajectory_accuracy(outputs: dict, inputs: dict, reference_outputs: dict) -> dict:
    return judge(
        outputs=outputs["messages"],
        reference_outputs=[
            HumanMessage(content=inputs["query"]),
            AIMessage(content=reference_outputs["reference_response"]),
        ],
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
        experiment_prefix="flight-llm-judge-v1",
        max_concurrency=2,
    )
    print(results.url)
