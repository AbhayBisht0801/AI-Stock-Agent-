import asyncio
from fastapi import FastAPI
from typing import Literal ,Annotated
from pydantic import BaseModel,Field
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)


from agent import main 

app=FastAPI()

class UserInput(BaseModel):
    user_request: Annotated[str,Field(...,description="User request to the AI agent")]
@app.post("/chat")
async def chat(user_input: UserInput):
    user_request = user_input.user_request
    result = await main(user_request)
    return result

    
