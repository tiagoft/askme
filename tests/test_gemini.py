import pytest
import askme.makequestions.api as api

from pydantic import BaseModel
from pydantic_ai import Agent

class HelloWorldResponse(BaseModel):
    message: str

def test_hello_world():
    model = api.make_gemini_model()
    prompt = "Respond with 'Hello, World!'"
    agent = Agent(
        model=model,
        output_type=HelloWorldResponse,
        instructions=prompt,
    )
    result = agent.run_sync("")
    assert result.output.message == "Hello, World!"
    