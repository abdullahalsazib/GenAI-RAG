import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


groqkey = os.getenv("GROQ_API_KEY")

load_dotenv()

model = ChatGroq(api_key=groqkey, model="llama-3.1-8b-instant")

store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_messges_history = RunnableWithMessageHistory(model, get_session_history)
config= {"configurable":{"session_id": "chat1"}}

# result=model.invoke([
#     HumanMessage(content="hey my name is Abdullah Al Sazib. i'm a software developer"),
#     AIMessage(content="Nice to meet you, Abdullah Al Sazib. What type of software development do you specialize in? Are you working on any exciting projects currently?"),
#     HumanMessage(content="what is my name, and what do i do?")
# ])

# parser = StrOutputParser()
# chain = model | parser

# print(chain.invoke(result.content))

result=with_messges_history.invoke([
    HumanMessage(content="my name is abdullahal sazib?"),AIMessage(content="As-salamu alaykum Abdullahal Sazib, it's nice to meet you. How can I assist you today? Would you like to discuss something specific or is it just a casual conversation?"),
    HumanMessage(content="what is my name and what do i do?")],
    config=config
)
# print("for chat_1",result.content)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistn"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model
chain.invoke({"messages": [HumanMessage(content="Hey my name is abdullah al sazib, i'm a software developer at dapplesoft.")]})
with_messges_history = RunnableWithMessageHistory(chain, get_session_history)
config= {"configurable":{"session_id": "chat3"}}

response = with_messges_history.invoke([HumanMessage(content="Hey my name is abdullah al sazib, i'm a software developer at dapplesoft.")], 
                                       config=config)
print(response)