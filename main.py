from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

pdfs_directory = 'pdfs/'
vector_db_directory = 'data'

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
model = OllamaLLM(model="llama3.2")

template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. If you don't know the answer, say that you don't know. 
Question: {question} 
Context: {context} 
Answer:
"""

def create_vector_store(file_path):
  print(f"Creating vector store from {file_path}")
  loader = PyPDFLoader(file_path)
  documents = loader.load()

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

def upload_pdf(file):
  with open(pdfs_directory + file.name, "wb") as f:
    f.write(file.getbuffer())
  create_vector_store(pdfs_directory + file.name)

def load_vector_store():
    print(f"Loading vector store...")
    vector_db = FAISS.load_local(vector_db_directory, embeddings, allow_dangerous_deserialization=True)  # Enable deserialization
    print(f"Vector store loaded successfully.")
    return vector_db

def check_if_vector_store_exists():
  try:
    vector_db = FAISS.load_local(vector_db_directory, embeddings, allow_dangerous_deserialization=True)  # Enable deserialization
    return True
  except Exception as e:
    print(f"Error loading vector store: {e}")
    return False

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

