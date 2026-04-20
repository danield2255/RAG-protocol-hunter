
from protocol_assistant import ProtocolAssistant

def main():
    assistant = ProtocolAssistant()
    rag_chain = assistant.setup_rag_chain()
    active_conversation = True
    while active_conversation:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            active_conversation = False
            print("Exiting the protocol assistant. Goodbye!")
        else:
            
            response = rag_chain.invoke(query)

            raw = response
            print("\nRAW RESPONSE for debugging:")
            print(raw)
            print("\n-------------")

            # Show which documents the retriever considered for this query (if available)
            try:
                if hasattr(assistant, "retriever") and assistant.retriever is not None:
                    docs_used = assistant.retriever.invoke(query)
                    print("\nDOCUMENTS USED (top {}):".format(len(docs_used)))
                    for i, d in enumerate(docs_used, start=1):
                        src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
                        preview = d.page_content.replace("\n", " ")[:300].strip()
                        if src:
                            print(f"{i}. source={src} — {preview}...")
                            print("-------")
                        else:
                            print(f"{i}. {preview}...")
                            print("$$$$$$$$$")
                else:
                    print("No retriever available on assistant to show documents.")
            except Exception as e:
                print("Could not fetch documents from retriever:", e)

            chain_of_thought = ""
            answer = ""
if __name__ == "__main__":
    main()