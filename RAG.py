# Wikipedia Query Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Arxiv Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# PDF Loader and FAISS Vector Store
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("OFF_Campus.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
Documents = text_splitter.split_documents(docs)

# Use Ollama embeddings (e.g., nomic-embed-text, llama3, etc.)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectordb = FAISS.from_documents(Documents, embeddings)
retriever = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retrieval_tool = create_retriever_tool(
    retriever=retriever,
    name="Document_Search",
    description="Useful to know information about Daphal Sanket Anil",
)

# --- OLLAMA MODEL SETUP ---
from langchain_community.chat_models import ChatOllama

# You can choose from: llama3, mistral, phi3, codellama, etc.
llm = ChatOllama(model="llama3", temperature=0.1)

# Prompt Template
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Agent Creation
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, [retrieval_tool], prompt=prompt)

# Agent Executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[retrieval_tool], verbose=True)

# --- Streamlit Interface ---
import streamlit as st

st.title("RAG Pipeline with Multiple Data Sources")
st.subheader("by - Sanket Daphal")
st.text("You can ask me about Sanket, or search seamlessly across Wikipedia and Arxiv — all in one place")

input_text = st.text_input("Enter your question here")

flag = False

if input_text:
    response = agent_executor.invoke({"input": input_text})
    st.write(response["output"])
    flag = True

if flag:
    st.write("Are you satisfied with the response?")
    if st.button("Yes"):
        st.write("Great! Glad you are satisfied with the response.")
    if st.button("No"):
        feedback = st.text_area("Please provide your feedback here")
        if feedback:
            st.write("Thank you for your feedback!")
