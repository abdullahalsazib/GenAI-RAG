#!/usr/bin/env python
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
# Load env variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="LangServe + Groq API",
    version="1.0",
    root_path="/groqapi",
    description="LangServe demo using Groq's Llama 3.1 model"
)


groq_model = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")

add_routes(
    app,
    groq_model,
    path="/groq"
)

add_routes(
    app,
    groq_model,
    path="/sazib",
)

# --- Add /translate route ---
prompt = ChatPromptTemplate.from_template(
    "Translate the following English text to {language}: {text}"
)
translation_chain = prompt | groq_model

add_routes(
    app,
    translation_chain,
    path="/translate"
)

# --- Run app ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
