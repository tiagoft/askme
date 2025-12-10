from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, UnexpectedModelBehavior
from pydantic import Field
import torch

class Answer(BaseModel):
    answer: bool = Field(description="True if 'yes', False if 'no'")

class GroundedAnswer(BaseModel):
    answer: bool = Field(description="True if 'yes', False if 'no'")
    evidence: str = Field(description="The evidence from the provided material that supports the answer.")

def ask_question_ollama(model,
                 instructions: str,
                 material: str,
                 output_type : BaseModel = Answer,
                retries=10,) -> AgentRunResult[Answer]:
    agent = Agent(
        model,
        output_type=output_type,
        retries=retries,
        instructions=instructions,
    )
    try:
        result = agent.run_sync(material)
    except UnexpectedModelBehavior as e:
        raise e
    
    return result
