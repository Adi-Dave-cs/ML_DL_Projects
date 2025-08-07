import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq  import ChatGroq
from langchain.chains import RetrievalQA
import tempfile
import os

# UI - Get user inputs
st.title("🤖 RAG App using GROQ + LangChain")

groq_key = st.text_input("🔑 Enter your GROQ API Key:", type="password")
input_method = st.selectbox("📄 Choose input type", ["Upload PDF/Text File", "Enter URL"])
query = st.text_input("💬 Ask a question:")

documents = []

# Handle user-uploaded file
if input_method == "Upload PDF/Text File":
    uploaded_file = st.file_uploader("Upload PDF or .txt", type=["pdf", "txt"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path)
        documents = loader.load()

# Handle user-provided URL
elif input_method == "Enter URL":
    url = st.text_input("🌐 Enter the webpage URL:")
    if url:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
        except Exception as e:
            st.error(f"Error loading URL: {e}")

# Proceed only if docs + query + API key
if groq_key and query and documents:
    with st.spinner("🔍 Processing your query..."):
        # Split docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Embed + Index
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # GROQ LLM
        llm = ChatGroq(api_key=groq_key, model="llama3-70b-8192")

        # RAG Chain
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        # Run
        result = chain.invoke({"query": query})


        # Show result
        st.subheader("✅ Answer:")
        st.write(result["result"])