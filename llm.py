import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Set page configuration
st.set_page_config(page_title="LLAMA 3 70B", page_icon=":melting_face:")

# Sidebar setup
with st.sidebar:
    st.title('LLAMA 3 70B Chat')
    conversational_memory_length = 10
    st.markdown('---')
    

# Initialize chat history and memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(k=conversational_memory_length)

# Create the chatbot instance
model = 'llama3-70b-8192'
groq_chat = ChatGroq(groq_api_key=st.secrets["groq_api_key"], model_name=model)
conversation = ConversationChain(llm=groq_chat, memory=st.session_state.memory)

# Display chat history
for message in st.session_state.chat_history:
    if message.get("role") == "user":
        with st.chat_message("user"):
            st.write(message.get("content"))
    elif message.get("role") == "assistant":
        with st.chat_message("assistant"):
            st.write(message.get("content"))

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(k=conversational_memory_length)

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to generate response
def generate_response(prompt_input):
    # Update memory with all previous messages
    for message in st.session_state.chat_history:
        st.session_state.memory.save_context(
            {'input': message.get('content')},
            {'output': ''} if message.get('role') == 'user' else {'output': message.get('content')}
        )
    
    response = conversation(prompt_input)
    return response['response']

# User-provided input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate a response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_input)
            st.write(response)
    
    # Append the assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
