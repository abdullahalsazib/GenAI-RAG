import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groqKey = os.getenv("GROQ_API_KEY")
model = ChatGroq(api_key=groqKey, model="llama-3.1-8b-instant")

messages = [
    SystemMessage(content="Translate the following from english to franch"),
    HumanMessage(content="Hello how are you")
]

result = model.invoke(messages)
parser = StrOutputParser()

# LCEL
chain = model | parser
lcelResponse=chain.invoke(messages)

# prompt template
generic_template="Translate english to {language}"
prompt=ChatPromptTemplate.from_messages(
    [("system", generic_template),("user","{text}")]
)

result2=prompt.invoke({"language":"Franch", "text": "Hello"})
result2.to_messages()
chain2 = prompt | model | parser
chain2_result=chain2.invoke({"language": "franch", "text": "Hello"}) 
print(chain2_result)