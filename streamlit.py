import streamlit as st
import main as main

# To run:
# st run .\st.py 

st.title("Chat with Documents with Ollama")

vector_store_exists = main.check_if_vector_store_exists()

if vector_store_exists:
  st.success(vector_store_exists)
else:
  st.warning("No vector store found. If there are documents in the resource directory, please recreate the vector store. Otherwise, upload a new document to create a vector store.")

question = st.chat_input("Ask a question or upload a new PDF", accept_file=True, file_type=["pdf", "html", "htm", "docx", "txt"])

if question and question["files"]:
  with st.status("Adding document to the vector store..."):
    main.upload_file(question["files"][0])
  st.toast("Ready to chat about your new document!", icon="✅")

if st.button("Recreate Vector Store"):
    with st.status("Recreating the vector store from all documents in the resource directory..."):
        main.create_vector_store_from_directory(main.resource_directory)
    st.toast("Vector store recreated successfully!", icon="✅")

if question and question["text"]:
  query = question["text"]
  st.chat_message("user").write(query)
  with st.status("Working on it..."):
      # Retrieve the most relevant documents from the vector store
      related_documents = main.retrieve_docs(query)
      answer = main.question_pdf(query, related_documents)
  st.chat_message("assistant").write(answer)
