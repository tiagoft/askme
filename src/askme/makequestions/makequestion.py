import pydantic_ai.models.openai as openai_models
from .api import azure_openai_client
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, UnexpectedModelBehavior
from askme.assets import rtp_prompts

class HypothesisAboutCollection(BaseModel):
    hypothesis: str


def crop_text_in_words(text: str, max_words: int) -> str:
    """Crop text to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    else:
        return ' '.join(words[:max_words])
    

def make_a_question_about_collection(
    collection: list[str],
    model: openai_models.OpenAIChatModel,
    retries: int = 10,
    blacklist: list[str] = [],
    max_words_per_text: int = 350,
) -> AgentRunResult[HypothesisAboutCollection]:
    """Generate a hypothesis about a collection of texts."""
    
    system_prompt = rtp_prompts['makequestions']['system_prompt']
    
    cropped_collection = [crop_text_in_words(text, max_words_per_text) for text in collection]

    user_prompt = f"Texts: {cropped_collection}"
    if blacklist is not None and len(blacklist) > 0:
        user_prompt += f"\nAvoid the following topics: {blacklist}\n"

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