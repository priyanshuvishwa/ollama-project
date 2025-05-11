import difflib
import os
import json
import logging
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain.vectorstores import VectorStoreRetriever  # Use VectorStoreRetriever instead
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import VectorStoreRetriever
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.chains import RunnablePassthrough

FAISS_DB_DIR = "faiss_index"  # Directory for FAISS index
RESPONSE_LOG_FILE = "aktu_cse_qa_data.json"
EMBEDDING_MODEL = "nomic-embed-text"  # Specify the embedding model
MODEL_NAME = "clara"  # Specify the model name for ChatOllama

def load_responses():
    """Load stored responses from a JSON file."""
    if os.path.exists(RESPONSE_LOG_FILE):
        with open(RESPONSE_LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def load_vector_db():
    """Load the FAISS vector database from disk."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    logging.info("✅ FAISS vector DB loaded from disk.")
    return vector_db

def find_similar_question(question, responses):
    """Find a similar question in the stored responses."""
    if not responses:
        return None
    similar_question = difflib.get_close_matches(question, responses.keys(), n=1, cutoff=0.8)
    return similar_question[0] if similar_question else None

def create_retriever(vector_db, llm):
    """Create a retriever using the vector database."""
    retriever = vector_db.as_retriever()
    logging.info("✅ Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create a chain to process the input question and generate a response."""
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
    if os.path.exists(os.path.join(FAISS_DB_DIR, "index.faiss")):
        vector_db = load_vector_db()
    else:
        documents = ingest_pdf(DOC_PATH)
        if documents is None:
            return
        chunks = split_documents(documents)
        vector_db = create_vector_db(chunks)

    # Load stored responses
    stored_responses = load_responses()

    # Example question
    question = "What is the syllabus of B.Tech 1st year?"

    # Check if a similar question has already been answered
    similar_question = find_similar_question(question, stored_responses)
    if similar_question:
        print("\n🤖 Clara says (from memory):")
        print(stored_responses[similar_question])
        return

    # Generate a new response
    docs = vector_db.as_retriever().get_relevant_documents("syllabus")
    print("\n🔍 Retrieved docs preview:")
    for doc in docs[:2]:
        print(doc.page_content[:200])

    # docs = vector_db.as_retriever().get_relevant_documents("syllabus")
    #     print("\n🔍 Retrieved docs preview:")
    #     for doc in docs[:2]:
    #         print(doc.page_content[:200])

    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db, llm)
    # Assuming create_chain is defined elsewhere
    chain = create_chain(retriever, llm)

    response = chain.invoke(input=question)
    print("\n🤖 Clara says:")
    print(response)

    # Save the new response
    save_response(question, response)

if __name__ == "__main__":
    main()