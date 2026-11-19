
import streamlit as st
from rag_engine import load_knowledge_base, ask_rag

st.title("🤖 College Info Assistant")

@st.cache_resource
def initialize_rag():
    return load_knowledge_base("01_College_Information.pdf")

embedding_model, chunks, index = initialize_rag()

api_key = st.secrets["GROQ_API_KEY"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Accept user question
question = st.chat_input("Ask about the college...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding answer..."):
            answer = ask_rag(
                question, embedding_model, chunks, index,
                api_key=api_key
            )
            st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
