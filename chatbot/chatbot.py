import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
groqkey = os.getenv("GROQ_API_KEY")

load_dotenv()

model = ChatGroq(api_key=groqkey, model="llama-3.1-8b-instant")
result=model.invoke([
    HumanMessage(content="hey my name is Abdullah Al Sazib. i'm a software developer"),
    AIMessage(content="Nice to meet you, Abdullah Al Sazib. What type of software development do you specialize in? Are you working on any exciting projects currently?"),
    HumanMessage(content="what is my name, and what do i do?")
])

parser = StrOutputParser()
chain = model | parser

print(chain.invoke(result.content))