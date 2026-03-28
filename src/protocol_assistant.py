from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from utils.rag_helpers import gather_docs, docs_to_embeddings, format_docs   


class ProtocolAssistant:
    def __init__(self):
        pass

    def get_pipeline(self):
        hf_pipeline = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            top_p=0.95,
            return_full_text=False
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm
        

    def setup_rag_chain(self):
        retriever = docs_to_embeddings(gather_docs())
        llm = self.get_pipeline()
        prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant for answering questions about system protocols, advice, and rules. All of your answers MUST be based on the retrieved documentation provided below.
            Your task is to read the retrieved documentation and use it to answer the user's question. You should also provide a brief chain of thought explaining how you arrived at your answer, based solely on the retrieved documentation.
            Use ONLY the retrieved documentation below to answer the single user question. If you don't know the answer, put exactly I don't know inside the <answer> element. Do not invent facts.

            Output Format Instructions (follow exactly):
            - Output MUST be exactly two XML elements and nothing else, in this exact order: a single <thought>...</thought> element immediately followed by a single <answer>...</answer> element.
            - Do NOT output examples, additional questions, labels, or any other text.
            - Do NOT generate follow-up questions or multiple Q/A pairs.
            - Keep <thought> concise (1-3 short sentences) and use only facts from the retrieved documentation.
            - Keep <answer> a single concise sentence that directly answers the question.
            - Do NOT generate additional questions
            - Do NOT repeat the prompt
            - Stop after giving the answer

            Documentation:
            {context}

            Question:
            {question}
            """)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            } | prompt | llm)
        
        # Ensure the chain of thought and answer are clearly separated in the response
        return rag_chain



