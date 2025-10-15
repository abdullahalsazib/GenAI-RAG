from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# data injetction - from the website scrape the data
loader = WebBaseLoader("https://docs.langchain.com/langsmith/evaluate-rag-tutorial")

# load
docs = loader.load()

# load llm
llm = ChatOllama(model="gemma3:1b")

# split the texts into documnet
text_spliter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
documents=text_spliter.split_documents(docs)

# embedding the documents into vector data
embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text:latest")

# vector db FAISS
vectorstoredb = FAISS.from_documents(documents, embeddings)

# query 
query = "Retrieval Augmented Generation (RAG)"

result = vectorstoredb.similarity_search(query)

# retrival chain
prompt = ChatPromptTemplate.from_template(
    """
answer the following question basd only on the provided context: 
<context>
{context}
</context>
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

document_chain