import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import tempfile
import os
import pandas as pd

output_csv = "C:\\Users\\mejethwa\\Downloads\\PDF_LLM-main\\PDF_LLM-main\\example_file.csv"
# Set up OpenAI API Key
openai_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI API key here",
    type="password")

if not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API key.")
 # Setting the API key for the current session

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Determine the file type and read accordingly
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            df = df.to_csv(output_csv, index=False)
        else:
            st.error("Unsupported file type")
            raise Exception("Unsupported file type")

        # Assuming data to be in the first column
        data = df[df.columns[0]].tolist()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
            retriever=vectorstore.as_retriever())

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Chat interface
        container = st.container()
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here :)", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = chain({"question": user_input, "chat_history": st.session_state['history']})
                st.session_state['history'].append((user_input, output["answer"]))

        response_container = st.container()
        with response_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state['history']):
                message(user_msg, is_user=True, key=f'user_{i}', avatar_style="big-smile")
                message(bot_msg, key=f'bot_{i}', avatar_style="thumbs")
    except Exception as e:
        st.error(f"An error occurred: {e}")


