import os
import json
import serpapi
from langsmith import traceable, get_current_run_tree
from dotenv import load_dotenv

load_dotenv()

client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"])

# @traceable(name="get_flights_info", run_type="tool")
def get_flights_info(origin: str, destination: str, date: str) -> str:
    """Get flights information for a given origin, destination, and date.

    Args:
        origin: Uppercase 3-letter IATA code (e.g. JFK) or a Google kgmid starting with /m or /g.
        destination: Uppercase 3-letter IATA code (e.g. LHR) or a Google kgmid starting with /m or /g.
        date: Departure date in YYYY-MM-DD format.
    """
    if os.environ.get("STUB_FLIGHTS") == "1":
        return json.dumps({
            "origin": origin,
            "destination": destination,
            "date": date,
            "flights": [],
        })
    try:
        results = client.search({
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "currency": "USD",
            "type": 2,
            "outbound_date": date,
        }).as_dict()

        rt = get_current_run_tree()
        if rt:
            add_total_time_taken_to_metadata(rt, results)
       

        return json.dumps(results)
    except Exception as e:
        return "Error: " + str(e)

@traceable(name="add_total_time_taken_to_metadata", run_type="tool")
def add_total_time_taken_to_metadata(rt, results):
    if rt:
        rt.metadata["total_time_taken"] = results.get("search_metadata", {}).get("total_time_taken")