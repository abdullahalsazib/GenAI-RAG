import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
# from langchain_core.runnables import RunnableLambda 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

documents = [
    Document(
        page_content="This is an example document that explains the purpose of the AI Test Project.",
        metadata={"source": "AI Test Project Overview"}
    ),
    Document(
        page_content="The project uses FastAPI as the backend, integrating LangChain for LLM-based interactions.",
        metadata={"source": "Architecture"}
    ),
    Document(
        page_content="Swagger UI automatically documents all API endpoints, but manual setup might be needed if using a proxy or custom prefix.",
        metadata={"source": "API Documentation"}
    ),
    Document(
        page_content="The AI Test Project demonstrates LLM-based Q&A through a Streamlit UI connected to the FastAPI backend.",
        metadata={"source": "Frontend Integration"}
    ),
    Document(
        page_content="Future improvements include adding database support, authentication, and LangSmith tracking for better debugging.",
        metadata={"source": "Future Enhancements"}
    )
]

documents = filter_complex_metadata(documents)

llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile")
embedding = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text:latest")

vectorstore = Chroma.from_documents(documents, embedding)

# result=vectorstore.similarity_search("what is fastapi")
# print(result)

# retriver = RunnableLambda(vectorstore.similarity_search).bind(k=1)
# result=retriver.batch(["llm", "langchain"])
# print(result)

retriver = vectorstore.as_retriever(
   search_type="similarity",
   search_kwargs={"k": 1},
)
# result=retriver.batch({"LLM", "Langsmith"})
# print(result)

messages = """
Answer the following question using the provided context only
{question}

context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", messages)])

rag_chain = {"context": retriver, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about fastapi")

print(response.content)