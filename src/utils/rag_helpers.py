import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def gather_docs(path = "resources"):
    docs = []
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            loader = TextLoader(os.path.join(path, file_name))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(docs)
    return docs


def docs_to_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


