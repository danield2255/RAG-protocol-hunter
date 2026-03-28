
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

            chain_of_thought = ""
            answer = ""
if __name__ == "__main__":
    main()