
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

st.title("RAG-based Q&A Demo")

# Sample knowledge base
docs = [
    "Python is a popular programming language.",
    "Hugging Face hosts transformer models.",
    "LangChain helps create LLM applications."
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(docs, embedding_model)
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    result = qa_chain.run(query)
    st.success(result)
