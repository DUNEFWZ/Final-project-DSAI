import google.generativeai as genai
import streamlit as st

# Set up sidebar
with st.sidebar:
    st.title('Celerates Final Project')
    if 'GEMINI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
    else:
        api_key = st.text_input('Enter Google Gemini API key:', type='password')
        if api_key:
            genai.configure(api_key=api_key)
            st.success('API key set!', icon='✅')
        else:
            st.warning('Please enter your Gemini API key!', icon='⚠️')

# Store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input logic
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Prepare chat history for context
        chat_history = [genai.ChatMessage(role=m["role"], content=m["content"])
                        for m in st.session_state.messages if m["role"] in {"user", "assistant"}]

        # Create chat session
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(prompt)

        full_response = response.text
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
