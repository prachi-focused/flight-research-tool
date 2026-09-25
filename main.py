from agent import agent

if __name__ == "__main__":
    context = """
    You are a helpful assistant that can help me find the best flights from New York to London on November 3rd, 2026.
    Provide the link to book the flight and the price of the flight.
    """
    result = agent.invoke(
        {"messages": 
            [
                {
                    "role": "user", 
                    "content": context
                }
            ],
        },
        config={
            "tags": ["flight_search"],
            "metadata": {
                "user_id": "111",
                "origin": "New York",
                "destination": "London",
                "date": "2026-11-03",
            },
        },
    )
    print("--------------------------------")
    print("-------AI Response:-------------")
    print("--------------------------------")
    print(result["messages"][-1].content)
    print("--------------------------------")
    print("-------------END----------------")
    print("--------------------------------")