import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Local LLM Chat", page_icon="🤖", layout="wide")

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #1e212b;
        color: #ffffff;
        border-radius: 10px;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API Base URL", value="http://localhost:11436/v1")
    model_name = st.text_input("Model Name", value="llama3")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Local LLM Chat")
st.caption(f"Connected to {api_url} | Model: {model_name}")

# Initialize OpenAI client
client = OpenAI(
    base_url=api_url,
    api_key="needed-but-not-used-by-local-llm"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response from the local LLM
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                temperature=temperature
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your local LLM server is running and the API BASE URL is correct.")
