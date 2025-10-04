#Wikipedia Query Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
  

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDRBNT_nsyuxMGekGKBJgTQvfFQ7wN054Y"

#%pip install --upgrade --quiet  langchain-google-genai

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://python.langchain.com/docs/introduction/?_gl=1*wpeed6*_gcl_au*NjE5MjI3MjMxLjE3NTk0MjA5MjM.*_ga*MTY2NTU3MzIwMS4xNzU5NDIwOTIz*_ga_47WX3HKKY2*czE3NTk1MTUxODIkbzYkZzEkdDE3NTk1MTUyMjAkajIyJGwwJGgw")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
Documents = text_splitter.split_documents(docs)

vectordb = FAISS.from_documents(Documents, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))

retriver = vectordb.as_retriever()
 

from langchain.tools.retriever import create_retriever_tool
retrieval_tool = create_retriever_tool(
    retriever=retriver, name="Document_Search", description="useful for when you need to find information about the python langchain"
)


#ARXIV Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper   

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
 

tools = [retrieval_tool, wiki, arxiv]

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDRBNT_nsyuxMGekGKBJgTQvfFQ7wN054Y"
from langchain_google_genai import ChatGoogleGenerativeAI  


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

#promt template
from langchain import hub

## get promt template from langchain hub
prompt = hub.pull("hwchase17/openai-functions-agent")
 

##Agent
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt=prompt)
 
 
 # To run agent we need agent executer
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#streamlit framework
import streamlit as st 
st.title("RAG Pipeline with Multiple Data Scources")
 
st.write("Ask me anything!")    
st.selectbox("Select a tool where you want to search", options=["Document_Search", "Wikipedia", "Arxiv"])
st.selectbox("Select a model", options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-turbo", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-turbo", "gemini-1.0-flash"])
input_text = st.text_input("Enter your question here")

flag = False
flag2 = False
if input_text:
    response = agent_executor.invoke({"input": input_text})
    st.write(response['output'])
    flag = True
 
if flag:
    st.slider("Are you satisfied with the answer?", min_value=1, max_value=10, step=1)
    flag2 = True
    if flag2:
        st.success("Thank you for your feedback!")
    

#agent_executor.invoke({"input": "what is langchain?"})