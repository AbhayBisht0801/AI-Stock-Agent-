
import streamlit as st
import asyncio
from agent import main

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot")
st.caption("Ask a question and get an AI-powered response.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt = st.chat_input("Type your message here...")

if prompt:
    # Display and store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(main(prompt))
                answer = result.get(
                    "final_answer",
                    "Sorry, I couldn't generate an answer."
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": str(answer)}
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Clear chat history
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
