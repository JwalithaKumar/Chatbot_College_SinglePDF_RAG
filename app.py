
import streamlit as st
from rag_engine import load_knowledge_base, ask_rag


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Jwalitha's AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Jwalitha's AI Assistant")

st.write(
    "Ask me about Jwalitha's professional experience, "
    "technical skills, training programs and achievements."
)

st.divider()


# 2. LOAD RAG KNOWLEDGE BASE

@st.cache_resource
def initialize_rag():
    return load_knowledge_base("Jwalithapro.pdf")


try:
    with st.spinner("Loading Jwalitha's profile..."):
        embedding_model, chunks, index = initialize_rag()

except Exception as e:
    st.error(f"Unable to load profile: {e}")
    st.stop()


# 3. GROQ API KEY

#api_key = os.getenv("GROQ_API_KEY")


try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("Please configure GROQ_API_KEY in Streamlit Secrets.")
    st.stop()


# 4. CHAT HISTORY

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 5. USER INPUT AND RAG RESPONSE

question = st.chat_input("Ask something about Jwalitha...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            try:
                answer = ask_rag(
                    question,
                    embedding_model,
                    chunks,
                    index,
                    api_key=api_key
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Unable to generate answer: {e}")


# 6. SIDEBAR


if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
