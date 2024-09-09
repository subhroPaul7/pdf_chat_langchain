# PDF ChatBot :books:

This readme file describes the PDF ChatBot, a Streamlit application that allows users to chat with a large language model (LLM) about the contents of a PDF document.

## How it Works

1. **Upload a PDF:** The user uploads a PDF document to the application.
2. **Process the PDF:** The application extracts text from the PDF and splits it into smaller chunks.
3. **Create Embeddings:** Embeddings are created for each text chunk using a sentence transformer model.
4. **Create Vector Store:** A vector store is created to store the embeddings and enable efficient retrieval of similar documents.
5. **Conversational Chain:** A conversational LLM chain is created using the LangChain library. This chain allows the user to have a conversation with the LLM about the PDF content.
6. **Suggestive Questions Chain:** A separate LLM chain is created to generate suggestive follow-up questions based on the user's queries.
7. **Chat Interface:** The application displays a chat interface where the user can ask questions about the PDF. The LLM will respond to the user's questions and the suggestive question chain will recommend additional questions the user might be interested in.

## Dependencies

* streamlit
* dotenv
* streamlit-chat
* langchain
* langchain-community
* transformers
* faiss

## Running the Application

1. Clone the repository containing the application code.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.


## Live link

https://pdf-chat-langchain.streamlit.app/