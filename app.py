
import streamlit as st
from rag_engine import load_knowledge_base, ask_rag

st.set_page_config(page_title="College Info Assistant")
st.title("🤖 College Info Assistant")
st.caption("Powered By StreamlitIO and GitHub")

@st.cache_resource
def initialize_rag():
    return load_knowledge_base("01_College_Information.pdf")

embedding_model, chunks, index = initialize_rag()

api_key = st.secrets["GROQ_API_KEY"]

question = st.chat_input("Ask about the college...")

if question:
    with st.spinner("Finding answer..."):
        answer = ask_rag(
            question, embedding_model, chunks, index,
            api_key=api_key
        )

    st.write("**Question:**", question)
    st.write("**Answer:**", answer)
