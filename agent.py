import os
from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tool import get_flights_info

load_dotenv()


def as_of() -> date:
    # Eval scripts set EVAL_TODAY before importing this module.
    raw = os.environ.get("EVAL_TODAY")
    if raw:
        return date.fromisoformat(raw)
    return date.today()


def build_system_prompt(today: date) -> str:
    return f"""
You are a helpful assistant that can help with flights information.

Today's date is {today.isoformat()}.

You must have all three of these before calling get_flights_info:
- origin (departure city or IATA code). If origin is a city, use the IATA code of the biggest airport in the city.
- destination (arrival city or IATA code). If destination is a city, use the IATA code of the biggest airport in the city.
- departure date in YYYY-MM-DD, after today's date

If any of those is missing, empty, or invalid, do not call any tool.
Reply in plain text explaining what is wrong with the user's prompt and what they still need to provide.
Never call get_flights_info with empty arguments.
"""


SYSTEM_PROMPT = build_system_prompt(as_of())

model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2,
)

agent = create_agent(
    model=model,
    tools=[get_flights_info],
    system_prompt=SYSTEM_PROMPT,
)