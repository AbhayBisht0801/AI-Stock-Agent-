import json

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
import sys 
# print("agent.py Python:", sys.executable)
# breakpoint()
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.types import Command
from common import print_report,stock_price_budget_check
import asyncio
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver,InMemorySaver
load_dotenv()
from utils_check import (
    AgentState, planner_node, tools_planner,route_after_planner, human_approval_node,route_after_approval,output_router,route_after_output
)
import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


client = MultiServerMCPClient(
    {
        "groww_mcp": {
            "transport": "sse",
            "url": "http://127.0.0.1:8000/sse",
        }
    }
)

import os

CONFIG_PATH = "config.json"

def discover_tool_description(tools) -> dict:
    """
    Introspects LangChain tool objects and builds a JSON-serializable
    description (name, description, JSON-schema args) for the planner prompt.
    """
    
    description = {}
    for t in tools:
        schema = convert_to_openai_tool(t)["function"]
        description[t.name] = {
            "description": t.description,
            "parameters": schema.get("parameters", {}),
        }
    return description
async def load_or_discover_tool_description(path: str = CONFIG_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [print_report]

    description = discover_tool_description(tools)

    with open(path, "w") as f:
        json.dump(description, f, indent=2)

    return description
async def build_graph():

    # Get MCP tools
    mcp_tools = await client.get_tools()

    tools = mcp_tools + [
        print_report,
        stock_price_budget_check
    ]

    # Create ToolNode
    tool_node = ToolNode(tools,messages_key='intermediate_step')

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("tools_planner", tools_planner)
    graph.add_node("planner", planner_node)
    graph.add_node("approval", human_approval_node)
    graph.add_node("tools", tool_node)
    graph.add_node('output_router',output_router)

    # -----------------------------------------
    # START → tools_planner
    # -----------------------------------------

    graph.add_edge(
        START,
        "tools_planner"
    )

    # -----------------------------------------
    # tools_planner → planner
    # -----------------------------------------

    graph.add_edge(
        "tools_planner",
        "planner"
    )

    # -----------------------------------------
    # planner → approval OR END
    # -----------------------------------------

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "approval": "approval",
            "end": END,
        }
    )

    # -----------------------------------------
    # approval → tools OR END
    # -----------------------------------------

    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # -----------------------------------------
    # tools → planner
    # -----------------------------------------
    graph.add_edge(
    "tools",
    "output_router"
)

# output_router → END OR planner
    graph.add_conditional_edges(
        "output_router",
        route_after_output,
        {
            "end": END,
            "planner": "planner",
        }
    )
    

    # Checkpointer required for interrupt/resume
    memory = MemorySaver()
    checkpointer = InMemorySaver()
    app = graph.compile(
        checkpointer=checkpointer
    )

    return app


# async def main():
#     chatbot = await build_graph()
#     user_request = "Want to sell my all Vodafone Idea Stock"

#     initial_state = {
#     "messages": [HumanMessage(content=user_request)],
#     "plan": [],
#     "current_step": 0,
#     "results": {},
#     "final_answer": "",
#     "resolution_error": None,
#     "aborted": False,
#     "Tool_Description": tool_description,
# }
    
    

#     # First invocation (will run until it hits the interrupt in prepare_tool)
#     result = await chatbot.ainvoke(initial_state, config=config)

#     # Check if the graph paused at an interrupt
#     # LangGraph exposes graph state tasks via get_state()
#     current_state = await chatbot.aget_state(config)
    
#     while current_state.tasks and any(task.interrupts for task in current_state.tasks):
#         # Extract the interrupt question/prompt if available
#         interrupt_val = current_state.tasks[0].interrupts[0].value
#         print(f"\n[INTERRUPT] {interrupt_val}")
        
#         # Get input manually from console
#         user_decision = input("Your choice (Yes/No): ").strip()
        
#         # Resume execution with the user's command
#         result = await chatbot.ainvoke(Command(resume=user_decision), config=config)
#         current_state = await chatbot.aget_state(config)

#     print("\n--- FINAL RESULT ---")
#     print(result.get("final_answer"))
#     return result
async def main(user_request: str):

    app = await build_graph()

   

    initial_state = {
        "input_message": [
            HumanMessage(content=user_request)
        ],
        "current_step": 0,
        "results": {},
        "final_answer": "",
        "previous_step": []
    }

    config = {
        "configurable": {
            "thread_id": "user_123"
        }
    }

    # First execution
    result = await app.ainvoke(
        initial_state,
        config=config
    )

    # Check whether graph is interrupted
    state = await app.aget_state(config)

    while state.tasks and any(
        task.interrupts for task in state.tasks
    ):

        interrupt_value = None

        for task in state.tasks:
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break

        print("\n========== APPROVAL REQUIRED ==========")
        print(interrupt_value)

        user_decision = input(
            "\nExecute this tool? (yes/no): "
        ).strip().lower()

        # Resume SAME thread
        result = await app.ainvoke(
            Command(resume=user_decision),
            config=config
        )

        state = await app.aget_state(config)

    print("\n========== FINAL RESULT ==========")
    print(result)

    return result

# async def main():

#     app = await build_graph()

#     # Get the graph
#     graph = app.get_graph()

#     # Generate PNG
#     png_data = graph.draw_mermaid_png()

#     # Save PNG
#     with open("agent_graph.png", "wb") as f:
#         f.write(png_data)

#     print("Graph saved successfully as: agent_graph.png")


if __name__ == "__main__":
    result=asyncio.run(main(user_request='What are the stocks available with me?'))
    breakpoint()

