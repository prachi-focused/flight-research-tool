import os
from agentevals.trajectory.llm import (
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    create_trajectory_llm_as_judge,
)
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from main import agent

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    judge=model,
)

def test_llm_as_judge_evaluator():
    result = agent.invoke({
        "messages": [HumanMessage(content="Suggest a flight from san francisco to tokyo for January 30th, 2027?")]
    })
    reference_output = [
        HumanMessage(content="Suggest a flight from san francisco to tokyo for January 30th, 2027?"),
        # ToolMessage(content="SF to Tokyo flight on January 30th, 2027", tool_call_id="123"),
        AIMessage(content="Listed are flights from San Francisco to Tokyo on January 30th, 2027"),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_output,
    )

    print("--------------------------------")
    print(evaluation)
    print("--------------------------------")
    for msg in result["messages"]:
        msg.pretty_print()
    print("--------------------------------")
    assert evaluation["score"] is True

    print("Test passed")

if __name__ == "__main__":
    test_llm_as_judge_evaluator()