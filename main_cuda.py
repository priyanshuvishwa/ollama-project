import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import torch
import json
from datetime import datetime
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = r"D:\myproject\myproject\data\Syllabus and Code of Conduct.pdf"
MODEL_NAME = "clara"
EMBEDDING_MODEL = "nomic-embed-text"
DATA_LOG_FILE = "training_data_log.json"
FAISS_DB_DIR = "faiss_index"


def check_cuda():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"✅ CUDA is available. GPU in use: {gpu_name}")
    else:
        logging.warning("⚠️ CUDA is NOT available. Running on CPU.")


def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("✅ PDF loaded successfully.")
        print(f"📄 Loaded {len(data)} documents from PDF.")
        print("Sample content:", data[0].page_content[:300])

        # Save metadata about the document load
        log_training_data(doc_path, len(data))

        return data
    else:
        logging.error(f"❌ PDF not found at: {doc_path}")
        return None


def log_training_data(file_path, num_documents):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "source_file": file_path,
        "num_documents": num_documents
    }
    if os.path.exists(DATA_LOG_FILE):
        with open(DATA_LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(DATA_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)
    logging.info("📁 Training data log updated.")


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = splitter.split_documents(documents)
    logging.info("✅ Documents split into chunks.")
    print(f"🧩 Number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i + 1}: {chunk.page_content[:100]}...")
    return chunks


def create_vector_db(chunks):
    ollama.pull(EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vector_db.save_local(FAISS_DB_DIR)
    logging.info("✅ New FAISS vector DB created and saved.")
    return vector_db


def load_vector_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    logging.info("✅ FAISS vector DB loaded from disk.")
    return vector_db


def create_retriever(vector_db, llm):
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
You are Clara, the Campus Learning Assistant. Your role is to help students with:

- Course information and syllabus

{question}"""
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=query_prompt
    )
    logging.info("✅ Retriever created.")
    return retriever


def create_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("✅ Chain created.")
    return chain


def main():
    check_cuda()  # Print CUDA availability

    # Load or create vector DB
    if os.path.exists(os.path.join(FAISS_DB_DIR, "index.faiss")):
        vector_db = load_vector_db()
    else:
        documents = ingest_pdf(DOC_PATH)
        if documents is None:
            return
        chunks = split_documents(documents)
        vector_db = create_vector_db(chunks)

    docs = vector_db.as_retriever().get_relevant_documents("syllabus")
    print("\n🔍 Retrieved docs preview:")
    for doc in docs[:2]:
        print(doc.page_content[:200])

    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)

    question = "What is the syllabus of B.Tech 1st year?"
    response = chain.invoke(input=question)
    print("\n🤖 Clara says:")
    print(response)


if __name__ == "__main__":
    main()
