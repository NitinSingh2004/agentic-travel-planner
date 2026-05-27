import os
from dotenv import load_dotenv
import serpapi

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

load_dotenv()

client = serpapi.Client(
    api_key=os.getenv("serp_api")
)


# ==========================
# FLIGHT TOOL
# ==========================

@tool
def search_flights_fn(
    arrival_id: str,
    departure_id: str,
    outbound_date: str
):
    """
    Search flights using airport codes and date.
    """

    results = client.search({
        "engine": "google_flights",
        "currency": "INR",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": "2",
    })

    flights = results.get("best_flights", [])

    if not flights:
        return "No flights found"

    output = f"Flights {departure_id} → {arrival_id}\n\n"

    for f in flights[:5]:

        price = f.get("price", "N/A")
        duration = f.get("total_duration", "N/A")

        segments = f.get("flights", [])

        if segments:
            first = segments[0]

            airline = first.get("airline", "N/A")
            dep = first.get("departure_airport", {}).get("id", "N/A")
            arr = first.get("arrival_airport", {}).get("id", "N/A")
        else:
            airline = dep = arr = "N/A"

        output += f"""
Airline: {airline}
Price: {price}
Route: {dep} → {arr}
Duration: {duration}

"""

    return output


# ==========================
# HOTEL TOOL
# ==========================

@tool
def get_hotel_deals(
    city_query: str,
    check_in: str,
    check_out: str
):
    """
    Search hotels.
    """

    params = {
        "engine": "google_hotels",
        "q": city_query,
        "check_in_date": check_in,
        "check_out_date": check_out,
        "adults": "2",
        "currency": "INR",
        "gl": "us",
        "hl": "en"
    }

    results = client.search(params)

    hotels = results.get("properties", [])

    if not hotels:
        return "No hotels found"

    all_hotels = []

    for hotel in hotels:

        all_hotels.append({
            "name": hotel.get("name"),
            "price_per_night":
                hotel.get("rate_per_night", {}).get("lowest"),
            "rating": hotel.get("overall_rating")
        })

    return str(all_hotels)


# ==========================
# AGENT FUNCTION
# ==========================

def run_travel_agent(chat_history):

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    tools = [
        search_flights_fn,
        get_hotel_deals
    ]

    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="""
You are an AI Travel Planner.

Flights:
- Need departure_id
- Need arrival_id
- Need outbound_date

Hotels:
- Need city_query
- Need check_in
- Need check_out

If information is missing, ask the user.
Never guess values.
"""
        )
    ]

    # Add chat history
    for msg in chat_history:

        if msg["role"] == "user":
            messages.append(
                HumanMessage(content=msg["content"])
            )

        elif msg["role"] == "assistant":
            messages.append(
                AIMessage(content=msg["content"])
            )

    # First LLM call
    response = llm_with_tools.invoke(messages)

    # No tool needed
    if not response.tool_calls:
        return response.content

    # IMPORTANT:
    # Add assistant response containing tool calls
    messages.append(response)

    # Execute tools
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        try:

            if tool_name == "search_flights_fn":

                result = search_flights_fn.invoke(
                    tool_args
                )

            elif tool_name == "get_hotel_deals":

                result = get_hotel_deals.invoke(
                    tool_args
                )

            else:

                result = f"Unknown tool: {tool_name}"

        except Exception as e:

            result = f"Tool Error: {str(e)}"

        # Add tool output properly
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    # Final LLM call with tool results
    final_response = llm_with_tools.invoke(messages)

    return final_response.content
