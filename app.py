import streamlit as st
from agent import run_agent

st.set_page_config(
    page_title="KBTCOE AI Agent",
    page_icon="🤖"
)

st.title("🤖 KBTCOE AI Agent")

st.write(
    "Ask a question or give the agent a task."
)

request = st.text_area(
    "What would you like the agent to do?"
)

if st.button("Run Agent"):

    if not request.strip():

        st.warning(
            "Please enter a request."
        )

    else:

        with st.spinner(
            "Agent is working..."
        ):

            try:

                result = run_agent(
                    request
                )

                st.subheader(
                    "Agent Result"
                )

                st.write(result)

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )