import os
import logging
import torch
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
CSV_PATH = r"D:\myproject\myproject\data\data.csv"  # <-- Use your CSV path here
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "clara-store"
PERSIST_DIR = "chroma_db"


def check_cuda():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"✅ CUDA is available. GPU in use: {gpu_name}")
    else:
        logging.warning("⚠️ CUDA is NOT available. Running on CPU.")


def ingest_csv(csv_path):
    if not os.path.exists(csv_path):
        logging.error(f"❌ CSV not found at: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    if df.empty:
        logging.error("❌ CSV is empty.")
        return None

    documents = []
    for i, row in df.iterrows():
        content = " ".join(str(v) for v in row.values if pd.notnull(v))
        documents.append(Document(page_content=content))

    logging.info(f"✅ CSV loaded and converted into {len(documents)} documents.")
    print("Sample content:", documents[0].page_content[:300])
    return documents


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
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=PERSIST_DIR,
        collection_name=VECTOR_STORE_NAME
    )
    logging.info("✅ New vector DB created and persisted.")
    return vector_db


def load_vector_db():
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=VECTOR_STORE_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    logging.info("✅ Existing vector DB loaded from disk.")
    return vector_db


def create_retriever(vector_db, llm):
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
You are Clara, the Campus Learning Assistant. Your role is to help students with:
- Course info
- Deadlines and schedules
- Study resources
- Academic services

Respond briefly, concisely, and only with relevant summary based on the question.

Question: {question}
"""
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=query_prompt
    )
    logging.info("✅ Retriever created.")
    return retriever


def create_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}\nRespond concisely and clearly."
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
    check_cuda()

    # Load or create vector DB
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vector_db = load_vector_db()
    else:
        documents = ingest_csv(CSV_PATH)
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
