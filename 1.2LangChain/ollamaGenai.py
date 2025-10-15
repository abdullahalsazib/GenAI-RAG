from langchain_community.llms.ollama import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Hey, you are a helpful expert AI agent."),
    ("human", "Question: {question}")
])

st.title("LangChain Demo App 🤖")
input_text = st.text_input("What’s the question on your mind?")

llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    # Stream output token by token
    with st.spinner("Thinking..."):
        st.write_stream(chain.stream({"question": input_text}))
