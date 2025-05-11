import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
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
DOC_PATH = r"D:\myproject\myproject\data\Syllabus and Code of Conduct.pdf"
MODEL_NAME = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "clara-store"
PERSIST_DIR = "chroma_db"


def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("✅ PDF loaded successfully.")
        print(f"📄 Loaded {len(data)} documents from PDF.")
        print("Sample content:", data[0].page_content[:300])
        return data
    else:
        logging.error(f"❌ PDF not found at: {doc_path}")
        return None


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
        template="""You are Clara, the Campus Learning Assistant! Help students with:
- Course info
- Deadlines and schedules
- Study resources
- Academic services

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
    # Load or create vector DB
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vector_db = load_vector_db()
    else:
        documents = ingest_pdf(DOC_PATH)
        if documents is None:
            return
        chunks = split_documents(documents)
        vector_db = create_vector_db(chunks)

    # ✅ This block is now correctly inside the function
    docs = vector_db.as_retriever().get_relevant_documents("syllabus")
    print("\n🔍 Retrieved docs preview:")
    for doc in docs[:2]:
        print(doc.page_content[:200])

    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)

    question = "What is the syllabus of B.Tech 1st year ?"
    response = chain.invoke(input=question)
    print("\n🤖 Clara says:")
    print(response)


# ✅ Now this will actually call main
if __name__ == "__main__":
    main()
