# RAG-With-Qdrant
DocumentGPT is a Streamlit-based application that allows users to upload various document types and interact with the content using conversational AI. The app leverages multiple technologies, including LangChain, Qdrant, and Google Generative AI, to process and understand document contents and provide accurate responses to user queries.
## Features
### File Upload and Processing:

  Supports PDF, DOCX, CSV, and TXT files.
  Uses PyMuPDFLoader, CSVLoader, Docx2txtLoader, and TextLoader for document parsing.
### Document Chunking:

  Utilizes RecursiveCharacterTextSplitter to split documents into manageable chunks for efficient processing.
### Embeddings and Vector Store:

Employs HuggingFaceEmbeddings to generate embeddings.
Uses Qdrant as the vector store for storing and retrieving document embeddings.
### Conversational AI:

Integrates Google Generative AI (Gemini-pro model) to provide accurate and context-aware responses.
Utilizes a custom prompt template to ensure precise and concise answers.
### Interactive Chat Interface:

Users can interact with the uploaded document content through a chat interface.
Streamlit chat interface for real-time Q&A.
## Installation
### Clone the Repository:



[git clone https://github.com/yourusername/document-gpt.git](https://github.com/Razamirxa/RAG-With-Qdrant.git)
cd document-gpt
### Create a Virtual Environment:



python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
### Install Dependencies:


pip install -r requirements.txt
### Set Up Environment Variables:

Create a .env file in the root directory with the following content:

GOOGLE_API_KEY=your_google_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
### Run the Application:

streamlit run main.py
## Usage
### Upload Documents:

Upload PDF, DOCX, CSV, or TXT files using the file uploader in the sidebar.
Click the "Process" button to process the uploaded files.
### Interact with the Documents:

Once processing is complete, use the chat input at the bottom of the page to ask questions about the document content.
View responses in the chat interface.
## Code Overview
main.py: The main script that sets up the Streamlit interface, handles file uploads, processes documents, and manages the chat interactions.
get_files_text function: Handles reading and extracting text from uploaded files.
get_text_chunks function: Splits extracted text into smaller chunks for efficient processing.
get_vectorstore function: Creates and manages the vector store using Qdrant.
rag function: Handles the retrieval and generation of answers using the specified AI model and prompt template.
## Dependencies
Streamlit: Web framework for creating interactive applications.
LangChain: Library for working with various language model tasks.
Qdrant: Vector store for managing and retrieving embeddings.
Google Generative AI: Model for generating context-aware responses.
dotenv: For managing environment variables.
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License.

## Acknowledgements
The LangChain team for their comprehensive libraries and tools.
The Qdrant team for their powerful vector store solution.
Google for their advanced AI models.
You can use this description in your README.md file for the GitHub repository. Make sure to replace placeholders (like yourusername, your_google_api_key, etc.) with actual values relevant to your project.
