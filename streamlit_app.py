import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pickle
import os
from PIL import Image
# img = Image.open('pdf.png')
# st.beta_set_page_config(page_title="PDF LLM", page_icon=img)
openai.api_key = st.secrets["OPENAI_API_KEY"]
  # Replace with your OpenAI API key

with st.sidebar:
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    st.title("📝 CG ChatGPT")

# st.title("📝 CG ChatGPT")
pdf = st.sidebar.file_uploader("Upload a PDF file", type=("pdf",))
if pdf:
    reader = PdfReader(pdf)

    # Read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # We need to split the text that we read into smaller chunks so that during information retrieval we don't hit the token size limits.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    store_name = pdf.name[:-4]
        
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vectorstore = pickle.load(f)
    else:
        # Embedding (OpenAI methods)
        embeddings = OpenAIEmbeddings()

        # Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

    if "messages" not in st.session_state:
        st.session_state.messages = []

# Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load the question-answering chain
    if query := st.chat_input("How may I help you?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

    
        docs = vectorstore.similarity_search(query=query, k=3)
        # st.write(docs)
        
        # OpenAI rank LNV process
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                response = chain.run(input_documents=docs, question=query)
                if response:
                    messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                
                full_response += response
                
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
          
#         #     print(cb)
#         # st.write("Bot:", response)
# import streamlit as st
# import openai
# from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
# import os
# from PIL import Image

# # Removed unused import 'pickle'

# # img = Image.open('pdf.png')
# # st.beta_set_page_config(page_title="PDF LLM", page_icon=img)

# openai.api_key = st.secrets["OPENAI_API_KEY"]
# # Replace with your OpenAI API key

# with st.sidebar:
#     # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
#     # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
#     st.title("📝 CG ChatGPT")

# # st.title("📝 CG ChatGPT")
# pdf = st.sidebar.file_uploader("Upload a PDF file", type=("pdf",))
# if pdf:
#     reader = PdfReader(pdf)

#     # Read data from the file and put them into a variable called raw_text
#     raw_text = ''
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             raw_text += text

#     # We need to split the text that we read into smaller chunks...
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     chunks = text_splitter.split_text(raw_text)
#     store_name = pdf.name[:-4]
        
#     # Handling FAISS object serialization
#     faiss_index_path = f"{store_name}_faiss.index"
#     if os.path.exists(faiss_index_path):
#         vectorstore = FAISS.load(faiss_index_path)
#     else:
#         # Embedding (OpenAI methods)
#         embeddings = OpenAIEmbeddings()
#         # Store the chunks part in db (vector)
#         vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
#         vectorstore.save(faiss_index_path)

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Load the question-answering chain
#     if query := st.chat_input("How may I help you?"):
#         st.session_state.messages.append({"role": "user", "content": query})
#         with st.chat_message("user"):
#             st.markdown(query)

#         docs = vectorstore.similarity_search(query=query, k=3)
#         # st.write(docs)
        
#         # OpenAI rank LNV process
#         llm = OpenAI(temperature=0)
#         chain = load_qa_chain(llm=llm, chain_type="stuff")
        
#         with get_openai_callback() as cb:
#             with st.chat_message("assistant"):
#                 message_placeholder = st.empty()
#                 full_response = ""
#                 response = chain.run(input_documents=docs, question=query)
#                 if response:
#                     messages=[
#                         {"role": m["role"], "content": m["content"]}
#                         for m in st.session_state.messages
#                     ],
#                     stream=True,
                    
#                     full_response += response
                    
#                     message_placeholder.markdown(full_response + "▌")
#                 message_placeholder.markdown(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})
          
            # print(cb)
            # st.write("Bot:", response)
