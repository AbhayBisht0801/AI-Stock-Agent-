import sys
import os
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils_check import planner_node

# load_dotenv(PROJECT_ROOT / ".env")
# os.environ["DEEPEVAL_EVAL_MODE"] = "system_one"

if not os.getenv("TYPESAFE_API_KEY"):
    raise RuntimeError("TYPESAFE_API_KEY is missing")

from deepeval import evaluate
from deepeval.metrics import (
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
)
from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallParams,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# If True, keys the planner added that never appear in ANY expected call for
# that tool are dropped before comparison (avoids penalising harmless
# optional/default arguments).
STRIP_EXTRA_ARGS = True

# Keys that are never meaningful for comparison.
IGNORED_KEYS = {"id", "tool_call_id", "type"}

Tool_correctness_prompt = """
You are an expert evaluator responsible for assessing the correctness of tool selection by an AI agent.

Evaluate whether the agent selected the appropriate tools based on the user's request, available tool descriptions, expected workflow, and provided context.

Follow these evaluation principles:

1. **Tool Relevance:** Every selected tool must be relevant to fulfilling the user's request.

2. **Tool Completeness:** Identify whether all necessary tools required to fulfill the request have been selected. Penalize missing tools when they are essential.

3. **Unnecessary Tool Selection:** Penalize tools that are irrelevant, redundant, or do not contribute to fulfilling the request.

4. **Workflow Correctness:** Evaluate whether the selected tools follow the logically correct sequence and dependencies, where applicable.

5. **Context Awareness:** Consider previous conversation steps, intermediate results, and already available information. Do not penalize the agent for avoiding redundant tool calls.

6. **Tool Availability:** Evaluate tool selection only against the tools made available to the agent. Do not expect unavailable tools to be selected.

7. **Intent Understanding:** Judge tool selection based on the user's actual intent. Distinguish between informational requests, analytical requests, and requests requiring actions.

8. **No Unnecessary Actions:** Do not expect action-oriented tools when the user only requests information, explanation, or analysis.

9. **Partial Correctness:** Give partial credit when the selected tools are relevant but the tool selection is incomplete or contains minor unnecessary calls.

10. **Evidence-Based Evaluation:** Base the assessment strictly on the user's request, available tool descriptions, expected tool calls, and supplied context. Do not invent requirements or assume tool capabilities.

Scoring Guidelines:
- 1.0: All necessary tools are selected, with no unnecessary tools.
- 0.8–0.99: Tool selection is substantially correct with minor omissions or redundancies.
- 0.5–0.79: Partially correct tool selection with significant omissions or unnecessary calls.
- 0.1–0.49: Mostly incorrect tool selection.
- 0.0: Completely incorrect tool selection or failure to select required tools.

An empty tool-call list is correct only when the user's request does not require any tool.

Evaluate the correctness of tool selection, not whether the tools successfully executed or produced the expected results.
"""
argument_correctness_prompt = """
You are an expert evaluator responsible for assessing the correctness of arguments passed by an AI agent to its selected tools.

Evaluate whether the arguments supplied to each tool are accurate, complete, relevant, and consistent with the tool's schema and the user's intent.

Follow these evaluation principles:

1. **Argument Accuracy:** Verify that each supplied argument has the correct value based on the user's request and available context.

2. **Required Arguments:** Ensure all required arguments specified in the tool description or schema are present.

3. **Optional Arguments:** Do not penalize omitted optional arguments unless they are necessary to fulfill the user's request.

4. **Parameter Names:** Argument names must match the tool's defined schema. Different parameter names must not be considered equivalent unless explicitly supported by the schema.

5. **Data Types:** Verify that argument values use the expected data types, such as strings, integers, floats, booleans, arrays, or objects.

6. **Context Consistency:** Arguments must be consistent with information provided in the current request, previous conversation, and intermediate tool results.

7. **Value Correctness:** Check that identifiers, dates, quantities, filters, configurations, and other values accurately reflect the user's intent.

8. **Missing Information:** Do not reward fabricated values or assumptions when the required information is unavailable. Evaluate whether the agent appropriately handles missing information.

9. **Extra Arguments:** Penalize unsupported or irrelevant arguments that could affect tool execution. Ignore harmless additional metadata only when the tool schema explicitly permits it.

10. **Tool-Specific Constraints:** Respect all constraints, dependencies, validation rules, and safety requirements defined in the tool descriptions.

11. **Argument Relevance:** Arguments must contribute to the intended operation and should not contain contradictory or unrelated information.

12. **Evidence-Based Evaluation:** Judge arguments only against the supplied user request, conversation context, intermediate results, and tool descriptions. Do not invent expected values or requirements.

Scoring Guidelines:
- 1.0: All arguments are correct, complete, and schema-compliant.
- 0.8–0.99: Arguments are substantially correct with minor non-critical issues.
- 0.5–0.79: Some arguments are correct, but significant inaccuracies or omissions exist.
- 0.1–0.49: Most arguments are incorrect, incomplete, or inconsistent.
- 0.0: Arguments are completely incorrect or fundamentally violate the tool requirements.

Evaluate argument correctness independently of whether the tool executed successfully or returned the expected result."""

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(PROJECT_ROOT / "config.json", encoding="utf-8") as f:
    tool_description = json.load(f)

golden_dataset_path = (
    PROJECT_ROOT / "golden_generation" / "planner_golden_dataset.json"
)
with open(golden_dataset_path, encoding="utf-8") as f:
    golden_dataset = json.load(f)
def extract_tools_call(planner_output):
    try:
        message = planner_output["intermediate_step"][0]
        kwargs = message.additional_kwargs

        # Legacy function_call format
        if "function_call" in kwargs:
            calls = [kwargs["function_call"]]

        # Modern tool_calls format
        elif "tool_calls" in kwargs:
            calls = [
                {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["args"]
                }
                for call in kwargs["tool_calls"]
            ]
        else:
            return []

        result = []

        for call in calls:
            arguments = call.get("arguments", {})

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            result.append(
                ToolCall(
                    name=call["name"],
                    input_parameters=arguments
                )
            )

        return result

    except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError):
        return []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def build_available_tools():
    """All tools from config.json, with descriptions, built once."""
    tools = []
    for name, spec in tool_description.items():
        if isinstance(spec, dict):
            desc = spec.get("description") or json.dumps(spec, default=str)
        else:
            desc = str(spec)
        tools.append(ToolCall(name=name, description=desc))
    return tools
available_tools = build_available_tools()
def run_evaluation():
    test_cases = []

    for index, item in enumerate(golden_dataset):
        try:
            inputs = item["input"]
            expected_output = item["expected_output"]

            user_request = inputs["user_request"]
            intermediate_step = inputs.get("intermediate_step", [])
            print('intermediate_step',intermediate_step)
            available_tool_names = inputs["tool_available"]

            print(f"\nEvaluating test case {index + 1}: {user_request}")

            planner_output = asyncio.run(planner_node({
                "input_message": [HumanMessage(content=user_request)],
                "intermediate_step": intermediate_step,
                "Tool_available": available_tool_names,  # check casing in planner_node
            }))

            actual_tools = extract_tools_call(planner_output)
            expected_tools = [
                ToolCall(
                    name=call["name"],
                    input_parameters=call.get("args", {})
                )
                for call in expected_output.get("tool_calls", [])
            ]
            

            test_case = LLMTestCase(
                # Plain user request so ArgumentCorrectness judges against it.
                input=user_request,
                actual_output=json.dumps(
                    {"tool_calls": actual_tools}, default=str
                ),
                tools_called=actual_tools,
                expected_tools=expected_tools ,
                additional_metadata={
                    "evaluation_instructions": Tool_correctness_prompt,
                    "intermediate_step":intermediate_step,
                    "available_tool_names": available_tool_names,
                },
            )
            
            test_cases.append(test_case)

            
            print('intermediate_step:', intermediate_step)
        except Exception as exc:
            print(f"Test case {index + 1} failed: {exc}")

    if not test_cases:
        print("No valid test cases to evaluate.")
        return
    

    

    tool_correctness = ToolCorrectnessMetric(
    threshold=0.8,
    available_tools=available_tools,
    evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    include_reason=True,
    eval_mode="llm",
    model="gpt-4o-mini",
    
)

    argument_correctness = ArgumentCorrectnessMetric(
        threshold=0.8,
        include_reason=True,
        eval_mode="llm",
        model="gpt-4o-mini",
        
    )

    evaluate(
        test_cases=test_cases,
        metrics=[tool_correctness, argument_correctness],
    )


if __name__ == "__main__":
    run_evaluation()