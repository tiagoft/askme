import pydantic_ai.models.openai as openai_models
from .api import azure_openai_client
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, UnexpectedModelBehavior
from askme.assets import rtp_prompts
class HypothesisAboutCollection(BaseModel):
    hypothesis: str


def make_a_question_about_collection(
    collection: list[str],
    model: openai_models.OpenAIChatModel,
    retries: int = 10,
    blacklist: list[str] = [],
) -> AgentRunResult[HypothesisAboutCollection]:
    """Generate a hypothesis about a collection of texts."""
    
    system_prompt = rtp_prompts['makequestions']['system_prompt']
     
    user_prompt = f"Texts: {collection}"
    if blacklist is not None and len(blacklist) > 0:
        prompt += f"\Avoid the following topics: {blacklist}\n"

    agent = Agent(
        model,
        output_type=HypothesisAboutCollection,
        retries=retries,
        instructions=system_prompt,
    )
    try:
        result = agent.run_sync(user_prompt)
    except UnexpectedModelBehavior as e:
        raise e
    
    return result