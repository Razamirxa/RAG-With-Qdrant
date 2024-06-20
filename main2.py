import os
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
import tempfile

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("DocumentGPT (Chat with your file)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        
        google_api_key = st.secrets["google_api_key"]
        qdrant_api_key = st.secrets["qdrant_api_key"]
        qdrant_url = st.secrets["qdrant_url"]
        
        if not google_api_key or not qdrant_api_key or not qdrant_url:
            st.info("Please add your API keys to continue.")
            st.stop()

        process = st.button("Process")
        if process:
            files_text = get_files_text(uploaded_files)
            st.write("File loaded...")
            text_chunks = get_text_chunks(files_text)
            st.write("File chunks created...")
            vectorstore = get_vectorstore(text_chunks, qdrant_api_key, qdrant_url)
            st.write("Vector Store Created...")
            st.session_state.conversation = vectorstore
            st.session_state.processComplete = True
            st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID

    if st.session_state.processComplete:
        input_query = st.chat_input("Ask Question about your files.")
        if input_query:
            response_text = rag(st.session_state.conversation, input_query, google_api_key)
            st.session_state.chat_history.append({"content": input_query, "is_user": True})
            st.session_state.chat_history.append({"content": response_text, "is_user": False})

            response_container = st.container()
            with response_container:
                for i, message_data in enumerate(st.session_state.chat_history):
                    message(message_data["content"], is_user=message_data["is_user"], key=str(i))

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == ".pdf":
            loader = PyMuPDFLoader(temp_file_path)
            pages = loader.load()
        elif file_extension == ".csv":
            loader = CSVLoader(file_path=temp_file_path)
            pages = loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
            pages = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
            pages = loader.load()
        else:
            st.error("Unsupported file format.")
            return ""

        for page in pages:
            text += page.page_content

        # Remove the temporary file
        os.remove(temp_file_path)
        
    return text

def get_vectorstore(text_chunks, qdrant_api_key, qdrant_url):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Qdrant.from_texts(texts=text_chunks, embedding=embeddings_model, collection_name="Machine_learning", url=qdrant_url, api_key=qdrant_api_key)
    return vectorstore

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    documents = [Document(page_content=text)]
    texts = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in texts]

def rag(vector_db, input_query, google_api_key):
    try:
        template = """You are an advanced AI assistant with expertise in document analysis. Your task is to provide precise and accurate answers to user questions based on the given context from uploaded documents. Follow these guidelines:

1. Use only the information provided in the context to answer the question.
2. If the context does not contain the information needed, respond with "I do not know based on the provided context."
3. Keep your answers clear, concise, and under 6 lines.
4. Do not provide information that is not present in the context.

Context:
{context}

Question:
{question}

Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )
        response = rag_chain.invoke(input_query)
        return response
    except Exception as ex:
        return str(ex)


if __name__ == '__main__':
    main()
