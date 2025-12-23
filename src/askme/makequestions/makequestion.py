import pydantic_ai.models.openai as openai_models
from .api import azure_openai_client
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, UnexpectedModelBehavior

class HypothesisAboutCollection(BaseModel):
    hypothesis: str


def make_a_question_about_collection(
    collection: list[str],
    model: openai_models.OpenAIChatModel,
    retries: int = 10,
) -> AgentRunResult[HypothesisAboutCollection]:
    """Generate a hypothesis about a collection of texts."""
    
    prompt = f"""Given the following collection of texts, generate a concise hypothesis
    that is true for half of the texts, but is not true for the other half.
    Texts:
    {collection}
    """

    agent = Agent(
        model,
        output_type=HypothesisAboutCollection,
        retries=retries,
        instructions=prompt,
    )
    try:
        result = agent.run_sync(collection)
    except UnexpectedModelBehavior as e:
        raise e
    
    return result