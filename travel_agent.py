import os
from dotenv import load_dotenv
import requests
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
ToolMessage
)

load_dotenv()

# client = serpapi.Client(
#     api_key=os.getenv("serp_api")
# )


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

    params = {
    "engine": "google_flights",
    "currency": "INR",
    "departure_id": departure_id,
    "arrival_id": arrival_id,
    "outbound_date": outbound_date,
    "type": "2",
    "api_key": os.getenv("serp_api")
           }

    results = GoogleSearch(params).get_dict()

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
    "api_key": os.getenv("serp_api")
}

    results = GoogleSearch(params).get_dict()
    # st.write(results)

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



@tool
def get_weather(city: str):
    """
    Get current weather for a city.

    Args:
        city: City name (e.g. London, Delhi, Paris)
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()
        # st.write(data)

        if response.status_code != 200:
            return f"Weather not found for {city}"

        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }

    except Exception as e:

        return f"Weather API Error: {str(e)}"


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
        get_hotel_deals,
        get_weather
    ]

    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="""
You are an AI Travel Planner specialized in helping users plan complete trips.

AVAILABLE TOOLS

1. Flight Search
   Required:
   - departure_id
   - arrival_id
   - outbound_date

2. Hotel Search
   Required:
   - city_query
   - check_in
   - check_out

3. Weather Lookup
   Required:
   - city

GENERAL RULES

- Always understand the user's travel intent before taking action.
- Never invent or guess dates, airport codes, cities, prices, weather data, or travel details.
- If required information is missing, ask clear follow-up questions.
- Ask only for information that is actually missing.
- Use tools only when all required parameters are available.
- Present information in a clean, organized, and user-friendly format.

FLIGHT SEARCH WORKFLOW

Before searching flights, ensure:
- departure location is known
- destination location is known
- travel date is known

If any information is missing, ask the user.

HOTEL SEARCH WORKFLOW

Before searching hotels, ensure:
- destination city is known
- check-in date is known
- check-out date is known

If any information is missing, ask the user.

WEATHER WORKFLOW

Use the weather tool when users:
- ask about weather
- ask what to pack
- ask about travel conditions
- ask the best time to visit a destination

TRIP PLANNING WORKFLOW

When a user wants help planning a trip:

Collect:
- departure city
- destination city
- travel dates
- budget (optional)

After sufficient information is available:

1. Search flights.
2. Search hotels.
3. Check destination weather when relevant.
4. Provide a travel summary.
5. Provide estimated trip costs when possible.
6. Suggest packing recommendations based on weather.
7. Highlight useful travel tips.

BUDGET PLANNING

If a user mentions a budget:

- Compare available travel options against the budget.
- Inform the user whether the trip appears within budget.
- If the budget is too low, suggest alternatives such as:
  - different dates
  - different hotels
  - nearby destinations
  - shorter stays

OUTPUT FORMAT

When presenting travel information:

✈️ Flights
- Airline
- Price
- Duration

🏨 Hotels
- Hotel Name
- Price Per Night
- Rating

🌤 Weather
- Temperature
- Conditions
- Packing Suggestions

💰 Budget Summary
- Flight Cost
- Hotel Cost
- Estimated Expenses
- Total Estimated Cost

Always provide concise, practical, and actionable travel advice.
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

    try:

        # First LLM call
        response = llm_with_tools.invoke(messages)
        # st.write(response.tool_calls)

        # If no tool call needed
        if not response.tool_calls:
            return response.content

        # Store assistant message containing tool calls
        messages.append(response)

        # Execute all requested tools
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            # st.write(tool_name)
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
                    # st.write(result)

                elif tool_name == "get_weather":
                    # st.write("working")
                    result = get_weather.invoke(
                        tool_args
                    )
                    # st.write(result)

                else:

                    result = (
                        f"Unknown tool requested: "
                        f"{tool_name}"
                    )

            except Exception as tool_error:

                result = (
                    f"Tool execution failed: "
                    f"{str(tool_error)}"
                )

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        # Final response after tool execution
        final_response = llm_with_tools.invoke(messages)

        return final_response.content

    except Exception as e:

        return f"Agent Error: {str(e)}"
