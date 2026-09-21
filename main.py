import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from typing import Annotated,TypedDict
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")



# client = MultiServerMCPClient(
#     {
#         "groww_mcp": {
#             "transport": "stdio",
#             'command':'python3',
#             "args": ["C:\\Users\\bisht\Downloads\expense_calculator\groww-python-mcp\main.py"],
#         }
#     }
# )
async def build_graph():
    
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(
        "create a summary of each tools and what does each tool do ?"
    )

    print(response)


asyncio.run(build_graph())