import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import asyncio
import json

from utils_check import tools_planner

import random 
DATASET_PATH = r"golden_generation\golden_dataset_long_term_investment.json"


def evaluate_tools(expected_tools, actual_tools):
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    common_tools = expected_set & actual_set

    return {
        "exact_match": expected_tools == actual_tools,

        "precision": (
            len(common_tools) / len(actual_tools)
            if actual_tools else 0
        ),

        "recall": (
            len(common_tools) / len(expected_tools)
            if expected_tools else 1
        ),
    }


async def evaluate_case(case):
    user_input = case["user_input"]
    expected_tools = case["expected_tools"]

    # Run planner
    actual_tools = await tools_planner({
        "input_message": user_input
    })

    evaluation = evaluate_tools(
        expected_tools=expected_tools,
        actual_tools=actual_tools["Tool_available"]
    )

    return {
        "input_message": user_input,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        **evaluation,
    }


async def main():

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    golden_dataset = dataset["golden_dataset"]

    # Select 10 random data points
    random_points = random.sample(
        golden_dataset,
        min(10, len(golden_dataset))
    )

    cases = random_points

    # Run ALL test cases concurrently
    results = await asyncio.gather(
        *(evaluate_case(case) for case in cases)
    )

    # Print results
    for result in results:
        print("\n" + "=" * 80)
        print("User:", result["input_message"])
        print("Expected:", result["expected_tools"])
        print("Actual:", result["actual_tools"])
        print("Exact Match:", result["exact_match"])
        print("Precision:", result["precision"])
        print("Recall:", result["recall"])

    # Overall metrics
    total = len(results)

    exact_matches = sum(
        result["exact_match"]
        for result in results
    )

    avg_precision = sum(
        result["precision"]
        for result in results
    ) / total

    avg_recall = sum(
        result["recall"]
        for result in results
    ) / total

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"Total cases       : {total}")
    print(f"Exact matches     : {exact_matches}")
    print(f"Exact accuracy    : {exact_matches / total:.2%}")
    print(f"Average precision : {avg_precision:.2%}")
    print(f"Average recall    : {avg_recall:.2%}")


if __name__ == "__main__":
    asyncio.run(main())