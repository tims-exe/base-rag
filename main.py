import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from vectordb import get_retriever, db_exists


# load vector db
while True:
    docs_folder = input("Enter the folder path containing documents (or type 'skip'): ")
    if docs_folder.lower() == "skip":
        if not db_exists():
            print("No existing database found. You have to specify documents to start chatting.")
            continue
        retriever = get_retriever()
        break
    else:
        if not os.path.exists(docs_folder):
            print(f"Folder '{docs_folder}' does not exist.")
            continue
        retriever = get_retriever(docs_folder)
        break

# Setup LLM
llm = OllamaLLM(model="llama3.2:latest", temperature=0.3)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def chat():
    print("="*20)
    print("RAG CHATBOT")
    print("="*20)
    
    chat_history = []

    while True:
        query = input("\n\nYou : ")
        if query.lower() == "exit" or query.lower() == "quit":
            break

        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        print(f"\n\nBot: {result['answer']}")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result['answer']))

if __name__ == "__main__":
    chat()