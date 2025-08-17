from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client= MultiServerMCPClient(
        {
            "maths":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio"
            },
            "weather":{
                "url":"http://localhost:8000/mcp",
                "transport":"streamable_http"
            }
        }
    )

    import os
    os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    
    model= ChatGroq(
        model="llama3-70b-8192",
        api_key=os.environ["GROQ_API_KEY"]
        )
    

    agent= create_react_agent(
        model,tools
    )

    math_response= await agent.ainvoke({"messages":[{"role":"user","content":"What is (2+2) x 3?"}]})

    print("Math response:",math_response['messages'][-1].content)

    weather_response= await agent.ainvoke({"messages":[{"role":"user","content":"What is the weather in Tokyo?"}]})

    print("Weather response:",weather_response['messages'][-1].content)

asyncio.run(main())