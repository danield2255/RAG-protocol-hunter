import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

docs = []
for file_name in os.listdir("resources"):
    if file_name.endswith(".txt"):
        loader = TextLoader(os.path.join("resources", file_name))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

hf_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


prompt = ChatPromptTemplate.from_template("""
Please answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs,
    "question": RunnablePassthrough()} | prompt | llm)

query = "What is the main topic of the documents?"
response = rag_chain.invoke(query)

print("RESPONSE:")
print(response)