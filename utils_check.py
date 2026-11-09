import json
import re
import uuid
from typing import Any, Optional,Annotated
from langchain_core.messages import BaseMessage,AIMessage
from operator import add
from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import os
from common import print_report,stock_price_budget_check

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
load_dotenv()
with open("config.json", "r") as f:
    tool_description = json.load(f)
client = MultiServerMCPClient(
    {
        "groww_mcp": {
            "transport": "sse",
            "url": "http://127.0.0.1:8000/sse",
        }
    }
)
# model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
model= ChatOpenAI(model="gpt-4o-mini")
DANGEROUS_TOOLS = [
    "groww_place_order",
    "groww_modify_order",
    "groww_cancel_order",
    "groww_create_smart_order",
    "groww_cancel_smart_order",
]


# ============================================================
# MODELS
# ============================================================

class PlanStep(BaseModel):
    step_id: str
    tool: str
    args_json: str  # JSON-encoded string, e.g. '{"symbol": "IDEA", "exchange": "NSE"}'
    depends_on: list[str]
class ExecutionPlan(BaseModel):
    steps: list[PlanStep]
class router_plan(BaseModel):
    fulfilled: bool
    final_message: str = ""
    intermediate_step:str=""

class AgentState(TypedDict):
    input_message:  list[BaseMessage]
    plan: list[PlanStep]
    current_step: int
    Tool_available: dict
    results: dict[str, Any]
    final_answer: str
    intermediate_step:Annotated[list[BaseMessage], add]
    aborted: bool
    router_fulfilled: bool
    Tool_executed: list[str]
class tool_result(BaseModel):
    tool: list[str]
planner_model = model.with_structured_output(ExecutionPlan, strict=False)
async def tools_planner(AgentState: dict) -> ExecutionPlan:
    """
    Generates a set of tools that needs to used with its details for planner node.
    """
    input_message=AgentState['input_message']
    tool_desc = {name: data["description"] for name, data in tool_description.items()}
    print(f"Tool Description: {json.dumps(tool_desc, indent=2)}")


    prompt=f"""
You are a tool planner for an AI Stock Agent.

Your job is just mention given the tool_name and description what are tools that would be usefull for this user request. Do not provide any other information or explanation.
User request:
{input_message}

Available tools:
{json.dumps(tool_desc, indent=2)}
Final output should be in json as below 
``` json 
{{'tools':['tool1','tool2']}}
```
"""

    result = await model.with_structured_output(tool_result).ainvoke(prompt)
    values = list(result.dict().values())[0]
    print(values)

    return {"Tool_available":values}
async def planner_node(state: AgentState):
    user_request = ""

    # Get original user request
    for message in reversed(state["input_message"]):
        if isinstance(message, HumanMessage):
            user_request = message.content
            print(f"User Request: {user_request}")
            break

    # Get latest execution history
    previous_messages = state.get("intermediate_step", [])

    if not previous_messages:
        previous_messages = "No Tool Executed Yet"
    else:
        previous_messages = previous_messages[-1]

    # Get tools
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [print_report, stock_price_budget_check]
    Tools_available={key: value
            for key, value in tool_description.items()
            if key in state['Tool_available']}
    tools = [
    tool
    for tool in tools
    if tool.name in state['Tool_available']
]
    # Bind tools to the model
    planner_model = model.bind_tools(tools)

    planner_prompt = f"""
You are the Tool Execution Agent for an AI stock analysis platform running inside a LangGraph workflow.

Your task is to determine and execute the next single tool action needed to fulfill the user’s request.

Execution Rules
Review the user request, execution history, previous tool results, and available tools.
Identify only the action required now . Never plan future steps or make multiple tool calls.
Understand the history results and use them to inform your next action. Do not repeat completed actions.
If a tool is required, generate exactly one native tool call supported by the available tools and valid arguments.
If no tool is needed, return a normal response. Do not call tools unnecessarily or repeat completed actions.
BUY Order Safety
Before any BUY order, a balance-check tool must have successfully executed.
Verify the returned balance is sufficient for the purchase.
If no balance check exists, call the balance-check tool first.
If funds are insufficient or cannot be confirmed, never place the BUY order.
Tool and Data Rules
Use only available tools and supported arguments.
Use values from the user request or previous tool results. Never fabricate arguments, IDs, or placeholder values.
Reuse previous results unless fresh information is explicitly required.
The exchange for all stock orders is NSE.
Completion Rule

If the request is already fulfilled or no further tool action is required, return a normal response without a tool call.

Execution Context

History: {previous_messages}

User Request: {user_request}

Available Tools: {Tools_available}
Output Requirements
Maximum one tool call per invocation.
No textual plans or explanations when a tool call is required.
Ensure the output is a valid native tool call for LangGraph's ToolNode.
Never invent tools, arguments, or future actions."""

    # Invoke model
    ai_message = await planner_model.ainvoke(planner_prompt)

    print("GENERATED AI MESSAGE:")
    print(ai_message)

    print("TOOL CALLS:")
    print(ai_message.tool_calls)

    return {
    "intermediate_step": [ai_message],
}

async def human_approval_node(state: AgentState):
    print('Failing here ')
    messages = state["intermediate_step"]

    if not messages:
        return {}

    last_message = messages[-1]

    # No tool call -> nothing to approve
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tool_call = last_message.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})

    # Only ask confirmation for dangerous tools
    if tool_name not in DANGEROUS_TOOLS:
        return {}
    prompt=f"""Given the tool name and its arguments, ask the user for confirmation to execute the tool. If the user confirms, return "yes". If the user denies, return "no". Do not provide any other information or explanation. The tool name is {tool_name} and its arguments are {tool_args}."""
    
    approval = interrupt({
        "type": "tool_confirmation",
        "tool": tool_name,
        "arguments": tool_args,
        "message": f"{model.invoke(prompt)}"
    })

    if approval != "yes":
        return {
            "aborted": True,
            "intermediate_step": [
                AIMessage(
                    content=f"Tool execution cancelled by user: {tool_name}"
                )
            ]
        }

    return {'Tool_executed':[tool_name]}
def route_after_planner(state: AgentState):

    messages = state["intermediate_step"]

    if not messages:
        return "end"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "approval"

    return "end"
def route_after_approval(state: AgentState):

    if state.get("aborted", False):
        return 'end'

    return "tools"
async def output_router(state: AgentState):
    prompt = f"""
You are the Output Router for an AI Stock Agent.

Your job is to determine whether the user's request has been fully completed based on:
1. The original user request
2. The result of the latest tool execution

You are NOT the final decision-maker for tool selection. Your job is to either:
- finalize the interaction when the request is complete, OR
- provide a concise execution instruction for the Planner Agent when more tools are required.

========================
USER REQUEST
========================
{state["input_message"]}

========================
LATEST TOOL OUTPUT
========================
{state["intermediate_step"][-1]}

========================
ROUTING RULES
========================
0. If there is authentication issue or any other issue that prevents the execution of the tool, return fulfilled=True and final_message=Sorry, I am unable to execute the requested action due to an authentication issue. Please check your credentials and try again.
1. Set `fulfilled=True` ONLY when the user's original request has been completely
   satisfied and no additional tool execution is required.
    Note:Note if the request is to buy stock and the user balance is 0 then stop as not transaction can be done so return fulfilled=True and return in final_message=Insufficient balance to buy 10 shares of RELIANCE.

2. Set `fulfilled=False` when:
   - The latest tool only provides information needed for a subsequent action.
   - The user's requested action has not yet been performed.
   - Another tool must be called to complete the original request.
   - The latest tool result is an intermediate result rather than the final result.

3. When `fulfilled=True`:
   - `final_message` must be a concise, user-facing answer.
   - `intermediate_step` must be EMPTY.
   - Do NOT request or suggest any additional tool execution.

4. When `fulfilled=False`:
   - `final_message` MUST be empty.
   - `intermediate_step` MUST contain a concise, explicit instruction describing
     what the Planner Agent needs to do next.
   - DO NOT write a user-facing answer.
   - DO NOT merely repeat the latest tool output.
   - DO NOT say things like "the user has 1 share".
   - Convert the tool result into the NEXT ACTION required to fulfill the
     ORIGINAL USER REQUEST.
   - Preserve all relevant information from the latest tool result that the
     next tool call will need.
   - The instruction must be actionable by the Planner Agent.

========================
IMPORTANT PRINCIPLE
========================

Always reason from the ORIGINAL USER REQUEST, not just from the latest tool output.

The latest tool output may only be an intermediate step.

For example:

User request:
"I want to sell all my stocks."

Latest tool output:
"The user has 1 share of Vodafone Idea (IDEA), average price ₹9.37."

This does NOT mean the request is fulfilled.

The correct routing result is:

fulfilled=False

intermediate_step:
"Based on the holdings result, the user has 1 share of Vodafone Idea (IDEA).
The next step is to sell all 1 share of IDEA as requested by the user."

The router must NOT generate:
"You have only 1 share of Vodafone Idea available for sale..."

That is a user-facing response and is incorrect because the requested sell action
has not yet been executed.

Another example:

User request:
"Buy 10 shares of RELIANCE."

Latest tool output:
"Current RELIANCE price is ₹2,500."


This is also NOT fulfilled.

Return:

fulfilled=False

intermediate_step:
"RELIANCE current price is ₹2,500. Proceed with the user's request to buy
10 shares of RELIANCE."

If the latest tool output confirms that the requested transaction was actually
executed, then:

fulfilled=True

final_message:
"Successfully bought 10 shares of RELIANCE."

intermediate_step:
[]

========================
OUTPUT FORMAT
========================

Return ONLY the structured output:

- fulfilled: boolean
- final_message: string
- intermediate_step: string

When fulfilled=False:
- final_message = ""
- intermediate_step = NEXT ACTION FOR PLANNER

When fulfilled=True:
- final_message = USER-FACING FINAL ANSWER
- intermediate_step = ""
"""

    result = await model.with_structured_output(router_plan).ainvoke(prompt)
    print({
        "router_fulfilled": result.fulfilled,
        "final_answer": result.final_message,
        'intermediate_step':[AIMessage(content=result.intermediate_step)]

    })
    
    return {
        "router_fulfilled": result.fulfilled,
        "final_answer": result.final_message,
        'intermediate_step':[AIMessage(content=result.intermediate_step)]

    }
   
def route_after_output(state: AgentState):
    if state["router_fulfilled"]:
        return "end"

    return "planner"
# async def main():
#     example_state = {
#         "input_message": "I want to sell all my Vodafone Idea stocks."
#     }

#     plan = await tools_planner(example_state)
#     print(plan)



if __name__ == "__main__":
    # Example usage
   
    plan = asyncio.run(main())
