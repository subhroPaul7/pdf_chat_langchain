import streamlit as st
#from dotenv import load_dotenv
#import numpy as np
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
#import spacy
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os
import tempfile
#import en_core_web_sm

#load_dotenv()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


def generate_suggestive_questions(vector_store):
    """
    This function creates a separate LLMChain specifically for suggesting questions.

    Args:
        vector_store: The vector store containing document embeddings (optional).

    Returns:
        RetrievalQAChain: The chain object for question suggestion.
    """

    llm  = ChatGroq(model="llama3-8b-8192", temperature=0.2)
    retriever = vector_store.as_retriever(score_threshold=0.7)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query")
    return chain

def suggest(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def create_conversational_chain(vector_store):
    llm  = ChatGroq(model="llama3-8b-8192", temperature=0.2)
    # Create llm

    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory
                                                 )
    
    return chain



def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history(chain1, chain2):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send ➤')

        if submit_button and user_input:
            sims =[]
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain1, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Felix")


def main():
    # Initialize session state
    initialize_session_state()
    # Initialize Streamlit
    st.title("PDF ChatBot :books:")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain1 = create_conversational_chain(vector_store)
        chain2 = generate_suggestive_questions(vector_store)
        
        display_chat_history(chain1, chain2)

if __name__ == "__main__":
    main()