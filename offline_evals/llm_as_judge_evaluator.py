import json
import os
import sys
from pathlib import Path

# Before the earliest future date in dataset.json (2026-11-01)
# and after the past-date example (2025-10-10).
os.environ["EVAL_TODAY"] = "2026-09-25"
os.environ.pop("STUB_FLIGHTS", None)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import agent

load_dotenv()

DATASET_NAME = "flight-evals"

judge_model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)


FLIGHT_CORRECTNESS_PROMPT = """You are evaluating a flight-search agent.

The agent can only look up one-way flights. It needs a departure city/airport, an arrival city/airport, and a future departure date. It searches with get_flights_info using IATA codes (or Google kgmids) and YYYY-MM-DD dates. It cannot do weather, hotels, or other non-flight tasks.

<Rubric>
  A correct response:
  - Stays in the flight-search domain
  - When origin, destination, and a future date are all present: presents flight options for that route and date (prices and booking links if the user asked for them). Wording may differ from the reference.
  - When origin, destination, or a future date is missing, empty, in the past, or too vague to search: does not invent flights. Explains what is wrong with the request and asks for the missing or invalid pieces.
  - When the user asks for something other than flights: refuses and says it only helps with flight information.
  - Does not claim tools or capabilities it does not have.

  Penalize:
  - Searching or listing flights when required fields are missing or invalid
  - Refusing or only asking follow-up questions when the request was already complete enough to search
  - Answering non-flight questions as if the agent had that capability
  - Wrong route or date relative to the user request and the reference
  - Fabricated flights, prices, or booking links that contradict a real search
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

Use the reference as the expected behavior for this request, not as required wording:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""

response_judge = create_llm_as_judge(
    prompt=FLIGHT_CORRECTNESS_PROMPT,
    judge=judge_model,
)

def invoke_agent(inputs: dict) -> dict:
    result = agent.invoke({"messages": [HumanMessage(content=inputs["query"])]})
    return {"messages": result["messages"]}

#  Judges if the AI response is correct based on the reference response
def response_accuracy(outputs: dict, inputs: dict, reference_outputs: dict) -> dict:
    return response_judge(
        inputs=inputs["query"],
        outputs=outputs["messages"][-1].content,
        reference_outputs=reference_outputs["reference_response"],
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
        evaluators=[response_accuracy],
        experiment_prefix="flight-llm-judge-v2",
        max_concurrency=2,
        metadata={"judge": "openai/gpt-oss-120b", "agent": "openai/gpt-oss-20b"},
    )
    print(results.url)
