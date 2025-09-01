from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory  # ✅ Updated import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
load_dotenv()
llm=ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)# temperatur means the more random the more creative the model is



store = {}

def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt=ChatPromptTemplate.from_messages([
    ("system", "you are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")])


chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history")



session_id ="user_123"

response1 =chain_with_history.invoke(
    {"input" : "Hello, how are you"},
    config={"configurable":{"session_id": session_id}
    }
)
print("AI:", response1.content)

response2=chain_with_history.invoke(
    {"input": "what is my next message you think it willbe"},
    config={"configurable":{"session_id": session_id}}
)

print("AI:", response2.content)

print("\nConversation History:")
for message in store[session_id].messages:
    print(f"{message.type}: {message.content}")