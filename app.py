import streamlit as st
from travel_agent import run_travel_agent


st.set_page_config(
    page_title="Travel AI Assistant",
    layout="centered"
)

st.title("✈️ Travel AI Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display old messages

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input(
    "Delhi to London next Tuesday"
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = run_travel_agent(
                    st.session_state.messages
                )

            except Exception as e:

                response = f"Error: {str(e)}"

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
