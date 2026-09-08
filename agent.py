import json
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tool import get_flights_info

load_dotenv()
SYSTEM_PROMPT = """
You are a helpful assistant that can help with flights information.
"""

model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

agent = create_agent(
    model=model,
    tools=[get_flights_info],
    system_prompt=SYSTEM_PROMPT,
)