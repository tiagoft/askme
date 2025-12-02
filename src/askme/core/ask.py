from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


class Answer(BaseModel):
    answer: bool

model = OpenAIChatModel(
    model_name='mistral:7b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),  
)

# Define the response type in the agent
agent = Agent(
    model,
    output_type=Answer,
    retries=10,
)

result = agent.run_sync("")

# result.data is now a validated CityInfo object, not just a string
print(f"Answer: {result.output.answer}")
print(f"Usage: {result.usage()}")
