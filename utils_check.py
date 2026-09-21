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
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
# model= ChatOpenAI(model="gpt-4o-mini")
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

    return {"Tool_available":{
        key: value
        for key, value in tool_description.items()
        if key in values
    }}
async def planner_node(state: AgentState):
    user_request = ""

    # Get original user request
    for message in reversed(state["input_message"]):
        if isinstance(message, HumanMessage):
            user_request = message.content
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

    # Bind tools to the model
    planner_model = model.bind_tools(tools)

    planner_prompt = f"""
You are the Tool Execution Agent for an AI stock analysis platform.

Your job is to decide and execute the NEXT SINGLE TOOL ACTION required
to fulfill the user's request.

You are running inside a LangGraph workflow.

The output of this node is sent directly to a LangGraph ToolNode.
Therefore, when a tool is required, you MUST generate a native tool call
using the tools available to you.

Do NOT generate a textual plan.
Do NOT describe the tool call in natural language.
Do NOT generate multiple tool calls.
Do NOT answer the user directly when a tool is required.

Your response must contain AT MOST ONE tool call.

--------------------------------------------------
EXECUTION MODEL
--------------------------------------------------

This agent operates iteratively.

On every invocation:

1. Read the original user request.
2. Read the latest execution history.
3. Inspect what tools have already executed and their results.
4. Determine what information/action is currently missing.
5. If another tool is required, call EXACTLY ONE tool.
6. If the request has already been fulfilled, do not call a tool.

The next invocation will happen after the selected tool has executed
and its result has been added to the execution history.

Never pre-plan future actions.

--------------------------------------------------
BUY ORDER RULES
--------------------------------------------------

Stock purchases are intended for long-term investment, not intraday trading.

Before placing ANY BUY order:

1. A balance-check tool MUST have successfully executed.
2. Inspect the balance-check result.
3. Determine whether sufficient funds are available.
4. Only if sufficient funds are confirmed may you call the BUY-order tool.
5. If funds are insufficient, NEVER call the BUY-order tool.

If a balance check has NOT been executed yet and the user wants to buy,
the balance-check tool is the immediate next action.

Do not assume that sufficient balance exists.

--------------------------------------------------
EXECUTION HISTORY RULES
--------------------------------------------------

Previous tool results are authoritative.

Reuse information already obtained from previous tool executions.

Do not call a tool again if its result already contains the information
required for the current action, unless fresh information is explicitly
required.

When a previous tool result provides a value required by another tool,
use that value.

--------------------------------------------------
TOOL RULES
--------------------------------------------------

Use ONLY tools that are actually available to you.

Never invent a tool name.

Use ONLY arguments supported by the selected tool.

Tool arguments must come from:

- The user's request, OR
- Previous tool results.

Never fabricate required values.

Never use placeholder values such as:

- "UNKNOWN"
- 0
- null
- ""
- fake IDs

unless that value is explicitly valid for the selected tool.

The exchange for stock orders is ALWAYS:

NSE

--------------------------------------------------
MINIMALITY
--------------------------------------------------

Only perform the action that is required RIGHT NOW.

Do not:

- Generate multiple tool calls
- Generate a future plan
- Perform unnecessary verification
- Repeat an already completed lookup
- Call tools simply because they are available
- Guess what the next invocation might need

--------------------------------------------------
COMPLETION
--------------------------------------------------

If the user's request can already be fulfilled using:

- The original user request
- Previous tool results
- Existing execution state

then DO NOT call another tool.

Instead, return a normal AI response indicating that no further
tool execution is required.

--------------------------------------------------
CURRENT EXECUTION HISTORY
--------------------------------------------------

{previous_messages}

--------------------------------------------------
USER REQUEST
--------------------------------------------------

{user_request}

--------------------------------------------------
AVAILABLE TOOLS
--------------------------------------------------

{state['Tool_available']}

--------------------------------------------------
FINAL REQUIREMENTS
--------------------------------------------------

- Maximum ONE tool call.
- Never generate multiple tool calls.
- Never generate a multi-step plan.
- Never invent tools.
- Never invent required arguments.
- Use previous tool results whenever possible.
- For BUY orders, balance validation is mandatory.
- If no tool is required, return a normal AI response.
- When a tool is required, generate a native tool call because your
  output will be passed directly to LangGraph's ToolNode.
"""

    # Invoke model
    ai_message = await planner_model.ainvoke(planner_prompt)

    print("GENERATED AI MESSAGE:")
    print(ai_message)

    print("TOOL CALLS:")
    print(ai_message.tool_calls)

    return {
        "intermediate_step": [ai_message],
        "current_step": 0,
        "results": {},
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

    approval = interrupt({
        "type": "tool_confirmation",
        "tool": tool_name,
        "arguments": tool_args,
        "message": f"Do you want to execute {tool_name}?"
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

    return {}
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

1. Set `fulfilled=True` ONLY when the user's original request has been completely
   satisfied and no additional tool execution is required.

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
