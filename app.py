import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader,DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema.runnable import Runnable
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.prompts import PromptTemplate

def load_documents_from_folder(folder_name="data"):
    folder_path= Path(os.getcwd()) / folder_name

    all_docs = []
    for file_path in folder_path.glob("*"):
            try:
                if file_path.suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix == ".docx":
                    loader = Docx2txtLoader(str(file_path))
                else:
                    continue
        
                docs= loader.load()
                all_docs.extend(docs)

            except Exception as e:
                st.warning(f"failed to load {file_path.name}: {e}")
    
    return all_docs


#scraping the website
def scrape_website():
    base_url="https://www.angelone.in/support"
    response= requests.get(base_url)
    soup= BeautifulSoup(response.text, "html.parser")
     

    page_texts = []
    for a in soup.find_all("a", href=True):
        href= a["href"]
        if href.startswith("/support"):
            full_url= f"https://www.angelone.in{href}"
            try:
                sub_response = requests.get(full_url)
                sub_soup = BeautifulSoup(sub_response.text,"html.parser" )
                text = sub_soup.get_text()
                page_texts.append(Document(page_content=text, metadata={"source": full_url}))
            except:
                pass
    return page_texts

def split_documents (docs, chunk_size=1000, chunk_overlap=0):
    splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_vectorstore(documents):

    embeddings = HuggingFaceEmbeddings(model_name= "intfloat/e5-small-v2")
    return FAISS.from_documents(documents, embeddings)


local_docs=load_documents_from_folder("data")
website_docs = scrape_website()
all_docs = local_docs+website_docs
chunks = split_documents(all_docs)
    


llm= ChatOpenAI (
    base_url = "https://6423-118-185-162-194.ngrok-free.app/v1",
    api_key="lm-studio",
    model_name="meta-llama-3.1-8b-instruct") 

vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True)


#streamlitUI 
st.set_page_config(page_title="Customer support", layout="centered")
st.title("Customer support chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("what can I help you with?")

if query:
    result = qa_chain.invoke({"query": query})
    answer = result["result"]


    st.write("Answer")

    if "I don't know" in answer or len (answer.strip())==0:
        st.warning("I don't know based on the provided PDFs")
    else:
        st.success(answer)
    
    st.write ("source documents")
    for doc in result["source_documents"]:
        st.write(f"-{doc.metadata.get('source','unknown')}")

    
        




