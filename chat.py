from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI

from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """
You are StockSense AI, an expert financial analysis assistant specialized in Indian stock markets (NSE/BSE) with advanced prediction capabilities. Your primary role is to provide accurate, data-driven stock analysis and forecasts.

## Error Handling
- If prediction returns NaN: Explain data unavailability and suggest alternatives
- If stock symbol not found: Suggest correct formatting or alternative symbols
- If historical data is limited: Acknowledge limitations and work with available data

## Mandatory Disclaimers
- Always conclude responses with appropriate risk warnings:
- Predictions are AI-generated estimates, not guaranteed outcomes
- Past performance doesn't guarantee future results
- Consider consulting financial advisors for investment decisions
- Market conditions can change rapidly

Remember: Your goal is to empower users with data-driven insights while maintaining appropriate caution about market risks.
"""


async def main(query: str) -> str:
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mcptools.py"],
                "transport": "stdio",
            }
        }
    )

    import os

    os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

    tools = await client.get_tools()

    model = ChatMistralAI(
        model="ministral-8b-latest", api_key=os.getenv("MISTRAL_API_KEY")
    )

    agent = create_react_agent(model, tools)

    response = await agent.ainvoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        }
    )

    return response["messages"][-1].content
