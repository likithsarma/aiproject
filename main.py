import streamlit as st
from langchain_helper import get_qa_chain,create_vector_db

st.title("travel alarm chatbot 🌱")
btn=st.button("create Knowledgebase")
if btn:
    create_vector_db()
question=st.text_input("question: ")
if question:
    chain=get_qa_chain()
    response=chain(question)

    st.header("Answer")
    st.write(response["result"])