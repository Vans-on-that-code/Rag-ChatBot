Customer Support RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot created with Streamlit, LangChain, FAISS, and HuggingFace Embeddings. The chatbot is programmed to give answers to user questions solely based on the content of PDF and DOCX documents it is trained on. If the information cannot be retrieved from the documents, the chatbot will answer that it doesn't know.

Project Overview

The chatbot imports documents, divides them into bite-sized pieces, creates vector embeddings, and indexes them in a FAISS index. When the user enters a query, the system fetches the most significant pieces and forwards them to a locally hosted language model (LLM) to generate a response.

Features
  Imports and processes PDF and DOCX documents
  Answers solely based on the content of the document
  Returns fallback message if answer is not available in the data
  Simple and interactive UI through Streamlit
  Completely offline working using LM Studio

Folder Structure
 \data/: Directory for storing all your input documents
main.py: Primary Python script with chatbot logic
 requirements.txt: Python dependencies
 README.md: Documentation

Prerequisites

1.Python 3.9 or above
2.LM Studio (installed on your machine)
3.A local LLM such as meta-llama-3.1-8b-instruct set up in LM Studio

Setup

1.Clone this repository using Git.
2.Create a virtual environment to separate dependencies.
3.Activate the virtual environment.
4.Install all dependencies with pip and the requirements.txt file.
5.Add your PDFs and DOCX files to the data/ folder.
6.Start LM Studio and make sure your selected model is running at http://localhost:1234/v1.

7. Run the Streamlit app.

Running the Application
 After the setup is finished:
  1.Open your terminal and cd into the project directory.
  2.Run the Streamlit app from the Streamlit command-line interface.
  3.Open the given local URL in a browser to use the chatbot.

Usage
1.Type a question into the text input field.
2.The chatbot will answer solely based on the documents within the data/ folder.
3.ource links or filenames from which the answers were obtained will be shown.
4.If the chatbot is unable to answer from the data, it will answer with a fallback message.

Notes
If website data scraping is not required, delete or comment out the web scraping part from the code.
Make sure the FAISS model and index are properly loaded upon every app start.

