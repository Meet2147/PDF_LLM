# #pip install streamlit langchain openai faiss-cpu tiktoken

# import streamlit as st
# from streamlit_chat import message
# from langchain-community.embeddings.openai import OpenAIEmbeddings
# from langchain-community.chat_models import ChatOpenAI
# from langchain-community.chains import ConversationalRetrievalChain
# from langchain-community.document_loaders.csv_loader import CSVLoader
# from langchain-community.vectorstores import FAISS
# import tempfile


# user_api_key = st.sidebar.text_input(
#     label="#### Your OpenAI API key 👇",
#     placeholder="Paste your openAI API key, sk-",
#     type="password")

# uploaded_file = st.sidebar.file_uploader("upload", type="csv")

# if uploaded_file :
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

#     loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
#     data = loader.load()

#     embeddings = OpenAIEmbeddings()
#     vectors = FAISS.from_documents(data, embeddings)

#     chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
#                                                                       retriever=vectors.as_retriever())

#     def conversational_chat(query):
        
#         result = chain({"question": query, "chat_history": st.session_state['history']})
#         st.session_state['history'].append((query, result["answer"]))
        
#         return result["answer"]
    
#     if 'history' not in st.session_state:
#         st.session_state['history'] = []

#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

#     if 'past' not in st.session_state:
#         st.session_state['past'] = ["Hey ! 👋"]
        
#     #container for the chat history
#     response_container = st.container()
#     #container for the user's text input
#     container = st.container()

#     with container:
#         with st.form(key='my_form', clear_on_submit=True):
            
#             user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
#             submit_button = st.form_submit_button(label='Send')
            
#         if submit_button and user_input:
#             output = conversational_chat(user_input)
            
#             st.session_state['past'].append(user_input)
#             st.session_state['generated'].append(output)

#     if st.session_state['generated']:
#         with response_container:
#             for i in range(len(st.session_state['generated'])):
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                 message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                
# #streamlit run tuto_chatbot_csv.py


# pip install streamlit langchain openai faiss-cpu tiktoken streamlit-chat

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
import tempfile

# Sidebar: API Key input
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI API key, e.g., sk-...",
    type="password"
)

# Sidebar: File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Check for API Key
if not user_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# Check for file upload
if uploaded_file:
    st.success(f"Uploaded file: `{uploaded_file.name}` successfully!")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load data from the CSV
    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    # Generate embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    vectors = FAISS.from_documents(data, embeddings)

    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=user_api_key),
        retriever=vectors.as_retriever()
    )

    # Function for chat interaction
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = [f"Hello! Ask me anything about `{uploaded_file.name}` 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Chat UI Containers
    response_container = st.container()
    input_container = st.container()

    # User input form
    with input_container:
        with st.form(key='input_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask something about your CSV data...", key="input")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

else:
    st.info("Upload a CSV file to get started!")
