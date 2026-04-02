import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field # Needed for strict tool schemas
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# --- 1. TOOL SCHEMAS (The Fix for the 400 Error) ---
class SearchInput(BaseModel):
    query: str = Field(description="The financial term or question to search for in the Tesla report")

class CalculatorInput(BaseModel):
    expression: str = Field(description="The math expression to evaluate, e.g., '25000 / 1.05'")

# --- 2. SETUP MODEL & DATA ---
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def get_retriever(file_path):
    embeddings = FastEmbedEmbeddings()
    persist_dir = "./chroma_db_tesla"
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings).as_retriever()
    loader = PyPDFLoader(file_path)
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
    return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir).as_retriever()

tesla_retriever = get_retriever("tesla.pdf")

# --- 3. DEFINE TOOLS WITH SCHEMAS ---
@tool(args_schema=SearchInput)
def search_tesla_report(query: str) -> str:
    """Search the Tesla Q4 2025 earnings report for financial data."""
    docs = tesla_retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Calculate math like '25000 / 1.05'. Use for growth/margins."""
    # Safety: eval is okay for personal learning, use a math library for production
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {e}"

web_search = TavilySearchResults(k=3)
tools = [search_tesla_report, calculator, web_search]
model_with_tools = model.bind_tools(tools)

# --- 4. SYSTEM PROMPT ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Financial Analyst Agent.
    - Use 'search_tesla_report' for any data inside the Tesla PDF.
    - Use 'tavily_search_results_json' for live data, competitor info, or anything NOT in the PDF.
    - Use 'calculator' only with ACTUAL numbers you found. Never use variable names.
    For comparisons, use the PDF for Tesla data and web search for external data."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# --- 5. UPDATED AGENT LOOP (Handles Groq's specific message flow) ---
def run_agent(user_input, history):
    chain = prompt | model_with_tools
    response = chain.invoke({"input": user_input, "chat_history": history})

    while response.tool_calls:
        history.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"--- Executing: {tool_name} with {tool_args} ---")
            tool_map = {"search_tesla_report": search_tesla_report, "calculator": calculator, "tavily_search_results_json": web_search}
            result = tool_map[tool_name].invoke(tool_args)
            history.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))

        response = model_with_tools.invoke(history)

    return response.content

# --- 6. CHAT LOOP ---
if __name__ == "__main__":
    chat_history = []
    print("\n--- Tesla Analyst Agent (Groq-Optimized 2026) ---")
    while True:
        user_q = input("User: ")
        if user_q.lower() in ["exit", "quit"]: break
        
        try:
            answer = run_agent(user_q, chat_history)
            print(f"\nAI: {answer}\n")
            
            # Persist for memory
            chat_history.append(HumanMessage(content=user_q))
            chat_history.append(AIMessage(content=answer))
        except Exception as e:
            print(f"\nError encountered: {e}")
            print("Tip: Ensure your PDF is named 'Tesla.pdf' and is in the same folder.")