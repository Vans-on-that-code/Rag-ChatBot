# Customer Support RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot application built using Streamlit, LangChain, FAISS, and HuggingFace Embeddings. The chatbot is designed to answer user queries strictly based on the content of PDF and DOCX documents provided by the user. If the chatbot cannot find the answer within the documents, it responds with "I don't know based on the provided PDFs."

## Project Description

This chatbot loads documents (PDF/DOCX), splits them into manageable chunks, converts them into embeddings using a HuggingFace model, and indexes them using FAISS. When a user asks a question, the chatbot retrieves the most relevant chunks and uses a language model (hosted locally via LM Studio) to generate an answer.

## Features

- Accepts PDF and DOCX documents as input
- Answers questions strictly from the content of the provided documents
- Declines to answer if the information is not found in the documents
- Simple and interactive web interface using Streamlit
- Works locally without requiring any API keys

## Folder Structure

```
RagChatbot/
├── data/                 # Folder to store your PDF and DOCX files
├── main.py               # Main application script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Requirements

- Python 3.9 or higher
- LM Studio (local model runner)
- LLM Model loaded in LM Studio (e.g., meta-llama-3.1-8b-instruct)

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### Step 2: Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On macOS/Linux
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Documents

Place your PDF and DOCX files in the `data/` folder. These documents will be used to generate answers.

### Step 5: Run LM Studio

Open LM Studio, start your model server, and ensure it runs at the base URL: `http://localhost:1234/v1`. The model used should be compatible with chat-style interactions (e.g., meta-llama-3.1-8b-instruct).

### Step 6: Run the Application

```bash
streamlit run main.py
```

If you want to share the app on your local network:

```bash
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Then access the app using `http://localhost:8501` or `http://<your-ip>:8501`

## Important Notes

- The chatbot only responds using content extracted from the provided documents. It will not generate hallucinated or unrelated answers.
- The phrase "I don't know based on the provided PDFs" is returned when the answer is not found in the indexed content.
- Website scraping is currently included in the code. If you only want the bot to answer from PDFs, comment out or remove the `scrape_website()` function and related logic.
