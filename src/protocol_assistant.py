import re
import torch
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from utils.rag_helpers import gather_docs, docs_to_embeddings, format_docs   


class ProtocolAssistant:
    def __init__(self):
        pass

    def get_pipeline(self):
        model_id = "meta-llama/Llama-3.1-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"stop": ["</answer>"]}
        )

        return llm
    
    def extract_answer(self, text):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "I don't know"
        

    def setup_rag_chain(self):
        retriever = docs_to_embeddings(gather_docs())
        llm = self.get_pipeline()

        prompt = ChatPromptTemplate.from_template("""
            You are a question-answering assistant.

            Answer the user's question using ONLY the provided documentation.

            Rules:
            - Return EXACTLY one answer.
            - Do NOT generate multiple questions or examples.
            - Do NOT add extra commentary.
            - If the answer is not in the documentation, return exactly: I don't know
            - Do NOT output anything after </answer>

            Output format:
            <answer>
            your answer here
            </answer>

            Documentation:
            {context}

            Question:
            {question}
            """)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            } | prompt | llm | self.extract_answer)
        
        # Ensure the chain of thought and answer are clearly separated in the response
        return rag_chain
    





