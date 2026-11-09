import json
import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import os

load_dotenv()
with open("config.json", "r") as f:
    tool_description = json.load(f)

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
    step_id: str = Field(description="Unique identifier for this step")
    tool: str = Field(description="Name of the tool to execute")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")
    depends_on: list[str] = Field(default_factory=list, description="IDs of previous steps this step depends on")


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]


class tool_result(BaseModel):
    tool: dict


class RepairedArgs(BaseModel):
    args: dict[str, Any]


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict):
    messages: list[BaseMessage]
    plan: list[PlanStep]
    current_step: int
    Tool_available: dict
    results: dict[str, Any]
    final_answer: str
    resolution_error: Optional[str]
    aborted: bool


planner_model = model.with_structured_output(ExecutionPlan)
repair_model = model.with_structured_output(RepairedArgs)


# ============================================================
# TOOL SELECTOR (currently unused in the graph)
# ============================================================

async def selector_node(state: AgentState):
    user_request = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_request = message.content
            break

    tool_desc = {name: data["description"] for name, data in tool_description.items()}

    prompt = f"""You are a Tool Selection Agent.

Your ONLY responsibility is to identify which available tools may be required to fulfill the user's request.
Keep in mind you are an AI Stock Analyser Tool.
Do NOT create an execution plan.
Do NOT execute tools.
Do NOT provide explanations.

Available Tools:
{tool_desc}

User Request:
{user_request}

Return ONLY valid JSON:
{{"tools": ["exact_tool_name_1", "exact_tool_name_2"]}}
"""
    result = await model.with_structured_output(tool_result).ainvoke(prompt)
    return {"Tool_available": result.dict()}


# ============================================================
# PLANNER NODE
# ============================================================

async def planner_node(state: AgentState):
    user_request = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_request = message.content
            break

    planner_prompt = f"""You are a Tool Execution Planning Agent for an AI stock analysis platform.

Your ONLY responsibility is to convert the user's request into a precise, minimal, executable tool plan in JSON.

CORE GUIDELINES:
1. **Investment Context:** Stock purchase orders are for long-term investments, never intraday trading.
2. **Balance Validation:** Always include a balance check step before any buy order step. The order step must depend on the output of the balance check.
3. **Minimality:** Include only necessary steps. Do not add speculative, redundant, or unrequested verification steps.
4. **No Execution:** Do NOT execute any tools or answer the user's request directly.

TOOL & ARGUMENT RULES:
- Use ONLY tools listed in Available Tools. Never invent or assume tool names.
- Use only parameters explicitly supported by the target tool.
- Populate arguments using values from the user's request, or reference previous step outputs using `$step_id.field` notation.
-The  exchange is always NSE for all stock orders.
- Never fabricate literal values (like "UNKNOWN" or 0) for required arguments that must come from a prior step.

DEPENDENCY RULES:
- `depends_on` must list all prerequisite step_ids (use [] if none).
- No circular or forward dependencies.

User Request:
{user_request}

Available Tools:
{tool_description}

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown fences, comments, or explanations.

{{
  "steps": [
    {{"step_id": "unique_step_name", "tool": "exact_tool_name", "args": {{}}, "depends_on": []}}
  ]
}}
"""
    plan = await planner_model.ainvoke(planner_prompt)
    print("GENERATED PLAN:")
    print(plan)

    return {
        "plan": plan.steps,
        "current_step": 0,
        "results": {},
        "resolution_error": None,
        "aborted": False,
    }


# ============================================================
# REFERENCE RESOLUTION (fast, deterministic path)
# ============================================================

REFERENCE_PATTERN = re.compile(r"^\$([a-zA-Z0-9_-]+)(.*)$")
TOKEN_PATTERN = re.compile(r"\.(?P<key>[a-zA-Z0-9_-]+)|\[(?P<index>\d+)\]")


def resolve_reference(value: Any, results: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    match = REFERENCE_PATTERN.match(value)
    if not match:
        return value

    step_id = match.group(1)
    remainder = match.group(2) or ""

    if step_id not in results:
        raise ValueError(f"Result for step '{step_id}' does not exist")

    result = results[step_id]

    for tok in TOKEN_PATTERN.finditer(remainder):
        if tok.group("key") is not None:
            key = tok.group("key")
            if isinstance(result, dict):
                if key not in result:
                    raise KeyError(f"Field '{key}' not found in step '{step_id}'")
                result = result[key]
            else:
                raise TypeError(f"Cannot access key '{key}' on {type(result)}")
        else:
            idx = int(tok.group("index"))
            if isinstance(result, list):
                if idx >= len(result):
                    raise IndexError(f"List index {idx} out of range for step '{step_id}'")
                result = result[idx]
            else:
                raise TypeError(f"Cannot index into {type(result)} on step '{step_id}'")

    return result


def resolve_arguments(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: resolve_arguments(val, results) for key, val in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(item, results) for item in value]
    return resolve_reference(value, results)


def _contains_unresolved_reference(value: Any) -> bool:
    """After resolution, check nothing that still looks like a '$ref' slipped through."""
    if isinstance(value, str):
        return bool(REFERENCE_PATTERN.match(value))
    if isinstance(value, dict):
        return any(_contains_unresolved_reference(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_unresolved_reference(v) for v in value)
    return False


# ============================================================
# PREPARE TOOL CALL (fast path)
# ============================================================

def _build_tool_call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}


def _confirm_dangerous_call(tool_call: dict):
    """Returns None if confirmed / not dangerous; returns an abort dict if the user declines."""
    if tool_call["name"] not in DANGEROUS_TOOLS:
        return None
    decision = interrupt(
        f"Are you sure you want to place/modify/cancel this order?\n"
        f"{json.dumps(tool_call['args'], indent=2, default=str)}\n(Yes/No)"
    )
    if isinstance(decision, str) and decision.lower() == "no":
        return {
            "messages": [AIMessage(content="Order placement/modification/cancellation aborted by user.", tool_calls=[])],
            "aborted": True,
            "resolution_error": None,
        }
    return None


async def prepare_tool_node(state: AgentState):
    step_index = state["current_step"]
    plan = state["plan"]

    if step_index >= len(plan):
        return {}

    step = plan[step_index]

    try:
        resolved_args = resolve_arguments(step.args, state["results"])
        if _contains_unresolved_reference(resolved_args):
            raise ValueError("Unresolved reference remained after resolution")
    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"ARG RESOLUTION FAILED for step '{step.step_id}': {e}")
        return {"resolution_error": str(e)}

    tool_call = _build_tool_call(step.tool, resolved_args)
    print("TOOL CALL:")
    print(tool_call)

    abort = _confirm_dangerous_call(tool_call)
    if abort:
        return abort

    return {"messages": [AIMessage(content="", tool_calls=[tool_call])], "resolution_error": None}


async def route_after_prepare(state: AgentState):
    if state.get("aborted"):
        return "aborted"
    if state.get("resolution_error"):
        return "needs_repair"
    return "ready"


# ============================================================
# REPAIR NODE (LLM path, only used when the fast path fails)
# ============================================================

async def repair_args_node(state: AgentState):
    step_index = state["current_step"]
    step = state["plan"][step_index]
    error = state.get("resolution_error", "")

    tool_schema = tool_description.get(step.tool, {})

    repair_prompt = f"""You are an Argument Repair Agent.

A planned tool call failed to resolve because a referenced field does not exist in the prior step's ACTUAL output.

Tool to call: {step.tool}
Tool parameter schema:
{json.dumps(tool_schema.get('parameters', {}), indent=2)}

Original (broken) arguments template:
{json.dumps(step.args, indent=2, default=str)}

Resolution error:
{error}

Actual results of all completed steps so far (ground truth — use ONLY real field names/values found here):
{json.dumps(state["results"], indent=2, default=str)}

Produce the CORRECT, fully concrete arguments object for calling '{step.tool}'.
Do not use '$step.field' placeholder syntax in your output — substitute real values directly.
If the required data genuinely is not present anywhere above, return the closest valid guess only if it is safe;
otherwise return an empty args object.

Return ONLY valid JSON:
{{"args": {{ ... }}}}
"""
    repaired = await repair_model.ainvoke(repair_prompt)

    tool_call = _build_tool_call(step.tool, repaired.args)
    print("REPAIRED TOOL CALL:")
    print(tool_call)

    abort = _confirm_dangerous_call(tool_call)
    if abort:
        return abort

    return {"messages": [AIMessage(content="", tool_calls=[tool_call])], "resolution_error": None}


async def route_after_repair(state: AgentState):
    return "aborted" if state.get("aborted") else "ready"


# ============================================================
# SAVE TOOL RESULT
# ============================================================

async def save_tool_result(state: AgentState):
    step_index = state["current_step"]
    step = state["plan"][step_index]

    tool_message = state["messages"][-1]
    content = tool_message.content

    if isinstance(content, str):
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = content
    else:
        result = content

    results = dict(state["results"])
    results[step.step_id] = result

    return {"results": results, "current_step": step_index + 1}


# ============================================================
# ROUTE AFTER TOOL
# ============================================================

async def route_after_tool(state: AgentState):
    if state["current_step"] < len(state["plan"]):
        return "next_tool"
    return "final"


# ============================================================
# FINAL ANSWER NODE
# ============================================================

async def final_answer_node(state: AgentState):
    if state.get("aborted"):
        abort_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content),
            "The action was aborted.",
        )
        return {"final_answer": abort_msg}

    user_request = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            user_request = message.content

    results_json = json.dumps(state["results"], indent=2, default=str)

    prompt = f"""
You are the final response agent.

Answer the user's original request.

User request:
{user_request}

The tools that were executed produced these results:
{results_json}

Rules:
- Answer directly.
- Do not mention internal planning.
- Do not mention step IDs.
- Do not invent information.
- Use the tool results as the source of truth.
"""
    response = await model.ainvoke(prompt)
    return {"final_answer": response.content}