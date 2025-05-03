import streamlit as st
import main as main

# To run:
# st run .\st.py 

st.title("Chat with PDFs with Ollama")

vendor_store_exists = main.check_if_vector_store_exists()

if vendor_store_exists:
  st.success("I still have information about the last PDF. You can start asking questions!")

question = st.chat_input("Ask a question or upload a new PDF", accept_file=True, file_type="pdf")

if question and question["files"]:
  with st.status("Transforming the PDF into a vector store..."):
    main.upload_pdf(question["files"][0])
  st.toast("Ready to chat about your PDF!", icon="✅")

if question and question["text"]:
  query = question["text"]
  st.chat_message("user").write(query)
  with st.status("Searching the vector store..."):
      # Retrieve the most relevant documents from the vector store
      related_documents = main.retrieve_docs(query)
      answer = main.question_pdf(query, related_documents)
  st.chat_message("assistant").write(answer)
