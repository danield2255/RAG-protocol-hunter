import re
import torch
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


from utils.rag_helpers import gather_docs, docs_to_embeddings, format_docs   
from utils.response_format import ResponseFormat


class ProtocolAssistant:
    def __init__(self):
        pass

    def get_pipeline(self):
        # model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        

        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )

        # pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     max_new_tokens=150,
        #     do_sample=False,
        #     temperature=0.0,
        #     return_full_text=False,
        #     repetition_penalty=1.1,
        #     early_stopping=True
        # )

        # llm = HuggingFacePipeline(
        #     pipeline=pipe,
        #     model_kwargs={"stop": ["</answer>"]}
        # )

        llm = ChatOllama(model="llama3.2:3b")

        return llm
    
    def extract_answer(self, text):
        # print("\nLLM RAW OUTPUT for debugging:")
        # print(text)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "I don't know"
        

    def setup_rag_chain(self):
        retriever = docs_to_embeddings(gather_docs())
        # keep the retriever on the instance so callers can inspect which docs were retrieved
        self.retriever = retriever
        llm = self.get_pipeline()

        structured_llm = llm.with_structured_output(ResponseFormat)

        prompt = ChatPromptTemplate.from_template("""
            You are a helpful question-answering assistant. 
            You will be provided with documentation and a question.
                                                  
            RULES:
            - Generate ONE SINGLE answer in a concise manner.
            - Answer the user's question using ONLY the provided documentation.
            - Do NOT generate new questions or examples.
            - If the answer is not explicitly stated in the documentation, say "I don't know".
                                                  
                                      
            Documentation:
            {context}

            Question:
            {question}
        """)

        
        

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            } | prompt | structured_llm )#| self.extract_answer)
            
        return rag_chain
    

