from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

PROMPT_IDENTIFIER = "flight-research-agent-prompt"

# Source of truth. Edit here, then `python -m agent_prompt push`.
SYSTEM_PROMPT = """
You are a helpful assistant that can help with flights information.

Today's date is {today}.

You must have all three of these before calling get_flights_info:
- origin (departure city or IATA code)
- destination (arrival city or IATA code)
- departure date in YYYY-MM-DD, after today's date

If any of those is missing, empty, or invalid, do not call any tool.
Reply in plain text explaining what is wrong with the user's prompt and what they still need to provide.
Never call get_flights_info with empty arguments.
If destination or origin  have multiple airports, provide all combinations of origin and destination airports.
"""

client = Client()


def push_prompt(commit_description: str = "") -> str:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT)])
    return client.push_prompt(
        PROMPT_IDENTIFIER,
        object=prompt,
        commit_description=commit_description,
    )


def pull_prompt() -> str:
    prompt = client.pull_prompt(PROMPT_IDENTIFIER)
    first = prompt.messages[0]
    if hasattr(first, "prompt"):
        return first.prompt.template
    return str(first.content)


def prompt_commit_id() -> str:
    commit = client.pull_prompt_commit(PROMPT_IDENTIFIER)
    return f"{PROMPT_IDENTIFIER}:{commit.commit_hash}"


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
    # Run: python -m agent_prompt push "commit description"
    if cmd == "push":
        description = sys.argv[2] if len(sys.argv) > 2 else ""
        print(push_prompt(description))
    # Run: python -m agent_prompt pull
    elif cmd == "pull":
        print(pull_prompt())
    # Run:  python -m agent_prompt commit
    elif cmd == "commit":
        print(prompt_commit_id())
    else:
        raise SystemExit("usage: python -m agent_prompt [push [description]|pull|commit]")
