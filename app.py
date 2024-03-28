import streamlit as st
#from dotenv import load_dotenv
import numpy as np
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import spacy
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
import tempfile

#load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
nlp = spacy.load("en_core_web_sm")

def generate_suggestive_questions(vector_store):
    """
    This function creates a separate LLMChain specifically for suggesting questions.

    Args:
        vector_store: The vector store containing document embeddings (optional).

    Returns:
        LLMChain: The chain object for question suggestion.
    """
    template = """You are an expert question generator. 
        You will be given a question. 
        Based on that, generate 3 short questions within 7 words which act as follow up to that question.
        Give me output as 3 bullets:
        - Question 1
        - Question 2
        - Question 3"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template ="{context}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    llm  = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2, max_tokens=128)
    retriever = vector_store.as_retriever(score_threshold=0.7)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": chat_prompt})
    return chain

def suggest(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def create_conversational_chain(vector_store):
    llm  = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
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

def display_chat_history(chain1, chain2, pages):
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
                for page in pages:
                    doc1 = nlp(output)
                    doc2 = nlp(page.page_content)
                    sims.append(doc2.similarity(doc1))
                num = np.argmax(sims)+1

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(f"{output} \n \nFrom page:{num}")

            # Generate suggestive questions based on the last query
            suggestive_questions = chain2.run({"query": user_input})
            if suggestive_questions:
                st.write("Here are some follow-up questions you might be interested in:")
                for question in suggestive_questions.strip().splitlines():
                    st.write(f"{question}")
            

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
                pages = loader.load()

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
        
        display_chat_history(chain1, chain2, pages)

if __name__ == "__main__":
    main()