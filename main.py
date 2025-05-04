import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredFileLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

resource_directory = 'resources/'
vector_db_directory = 'data'

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant that answers questions. Using the following retrieved information, answer the question provided. If you don't know the answer, say that you don't know. 
Question: {question} 
Context: {context} 
"""

def create_vector_store_from_directory(directory):
    print(f"Creating vector store from files in {directory}")

    # Log the date and time the function is run
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("vector_store_log.txt", "a") as log_file:
        log_file.write(f"Vector store created on: {current_time}\n")

    documents = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith('.docx') or file_name.endswith('.doc'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_name.endswith('.txt'):
            loader = UnstructuredFileLoader(file_path)
        elif file_name.endswith('.html') or file_name.endswith('.htm'):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            print(f"Unsupported file type: {file_name}, skipping.")
            continue

        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300, 
        add_start_index=True
    )

    chunked_docs = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunked_docs, embeddings)

    # Save the vector database locally
    vector_db.save_local(vector_db_directory)
    print(f"Vector store created and saved successfully.")

def upload_file(file):
  with open(resource_directory + file.name, "wb") as f:
    f.write(file.getbuffer())
  add_document_to_vector_store(resource_directory + file.name)

def load_vector_store():
    print(f"Loading vector store...")
    vector_db = FAISS.load_local(vector_db_directory, embeddings, allow_dangerous_deserialization=True)  # Enable deserialization
    print(f"Vector store loaded successfully.")
    return vector_db

def check_if_vector_store_exists():
    try:
        if os.path.exists("vector_store_log.txt"):
            with open("vector_store_log.txt", "r") as log_file:
                return log_file.read()
        else:
            return None
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def retrieve_docs(query, k=4): # k = number of documents to retrieve
  db = load_vector_store()
  print()
  print("==========================")
  print(f"Query: {query}")
  print(f"Vector db search results: {db.similarity_search(query)}")
  return db.similarity_search(query, k)

def question_pdf(question, documents):
  context = "\n\n".join([doc.page_content for doc in documents])
  prompt = ChatPromptTemplate.from_template(template)
  chain = prompt | model

  return chain.invoke({"question": question, "context": context})

def add_document_to_vector_store(file_path):
    print(f"Adding document {file_path} to the vector store")

    # Determine the loader based on file type
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = UnstructuredFileLoader(file_path)
    elif file_path.endswith('.html') or file_path.endswith('.htm'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        print(f"Unsupported file type: {file_path}, skipping.")
        return

    # Load the document
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300, 
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)

    # Load the existing vector store
    try:
        vector_db = FAISS.load_local(vector_db_directory, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Creating a new vector store instead.")
        vector_db = FAISS.from_documents([], embeddings)

    # Add the new document to the vector store
    vector_db.add_documents(chunked_docs)

    # Save the updated vector store
    vector_db.save_local(vector_db_directory)
    print(f"Document {file_path} added to the vector store successfully.")

