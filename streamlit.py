import streamlit as st
import main as main

# To run:
# st run .\st.py 

st.title("Chat with PDFs with Ollama")

pdf_processed = False
vendor_store_exists = main.check_if_vector_store_exists()

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False )

if uploaded_file:
  if not pdf_processed:
    with st.status("Transforming the PDF into a vector store..."):
      main.upload_pdf(uploaded_file)
      pdf_processed = True
    st.toast("Ready to chat about your PDF!", icon="✅")

if vendor_store_exists:
  st.success("I still have information about the last PDF. You can start asking questions!")
  
question = st.chat_input("Ask a question about this PDF")

if question:
  st.chat_message("user").write(question)
  with st.status("Searching the vector store..."):
      # Retrieve the most relevant documents from the vector store
      related_documents = main.retrieve_docs(question)
      answer = main.question_pdf(question, related_documents)
  st.chat_message("assistant").write(answer)
