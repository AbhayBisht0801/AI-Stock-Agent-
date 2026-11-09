from utils_check import planner_node,tools_planner
import asyncio
from langchain.messages import HumanMessage
async def main():
    return  await planner_node( {
      "input_message": [HumanMessage(content="Buy 10 shares of TCS for long-term investment.")],
      "Tool_available": ["groww_get_balance", "stock_price_budget_check","groww_place_order"],
      "intermediate_step":["Balance check successful. Available balance is Rs. 1000. Estimated total cost for the requested shares is Rs. 35000. Funds are insufficient. Do not place the order."]
    })
    # return await tools_planner({
    #     "input_message": "Analyze Reliance stock performance for the last 6 months."})
if __name__ == "__main__":
    result = asyncio.run(main())
    breakpoint()

