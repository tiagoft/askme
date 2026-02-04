import pydantic_ai.models.openai as openai_models
from .api import make_azure_model, make_ollama_model, azure_openai_client
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, UnexpectedModelBehavior
from askme.assets import rtp_prompts
from askme.config.config import MakeQuestionsConfig, config_factory
import shelve
from pathlib import Path

class HypothesisAboutCollection(BaseModel):
    hypothesis: str

class QuestionMaker:
    def __init__(self, config: MakeQuestionsConfig | None = None):
        if config is None:
            config = config_factory(MakeQuestionsConfig)

        self.config = config
        assert isinstance(self.config, MakeQuestionsConfig)
        # Initialize LLM model
        if self.config.model_name.startswith('gpt-4o'):
            self.llm_model = make_azure_model(self.config.model_name)
        else:
            self.llm_model = make_ollama_model(self.config.model_name)
        self.cache_fn = Path(self.config.cache).expanduser()
        
    def __call__(self, collection: list[str]) -> AgentRunResult[HypothesisAboutCollection]:
        
        return make_a_question_about_collection(
            collection,
            model=self.llm_model,
            retries=self.config.retries,
            blacklist=self.config.blacklist,
            max_words_per_text=self.config.max_words_per_text,
            cache_fn=self.cache_fn,
        )

def crop_text_in_words(text: str, max_words: int) -> str:
    """Crop text to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    else:
        return ' '.join(words[:max_words])
 
def make_a_question_about_split(
    collection_a: list[str],
    collection_b: list[str],
    model: openai_models.OpenAIChatModel,
    retries: int = 10,
    max_words_per_text: int = 350,
    blacklist: list[str] = [],
    cache_fn: str | None = None,
) -> AgentRunResult[HypothesisAboutCollection]:
    """Generate a hypothesis about the difference between two collections of texts."""
    
    system_prompt = rtp_prompts['splitquestion']['system_prompt']
    
    cropped_collection_a = [crop_text_in_words(text, max_words_per_text) for text in collection_a]
    cropped_collection_b = [crop_text_in_words(text, max_words_per_text) for text in collection_b]

    user_prompt = f"Collection A: {cropped_collection_a}\n\nCollection B: {cropped_collection_b}"
    
    if blacklist is not None and len(blacklist) > 0:
        user_prompt += f"\nAvoid the following topics: {blacklist}\n"
    
    if cache_fn is not None:
        prompt_hash = hash((system_prompt, user_prompt))
        cache = shelve.open(cache_fn)
        if prompt_hash in cache:
            cached_result = cache[prompt_hash]
            cache.close()
            return cached_result
        
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


def make_a_question_about_collection(
    collection: list[str],
    model: openai_models.OpenAIChatModel,
    retries: int = 10,
    blacklist: list[str] = [],
    max_words_per_text: int = 350,
    cache_fn: str | None = None,
) -> AgentRunResult[HypothesisAboutCollection]:
    """Generate a hypothesis about a collection of texts."""
    
    system_prompt = rtp_prompts['makequestions']['system_prompt']
    
    cropped_collection = [crop_text_in_words(text, max_words_per_text) for text in collection]

    user_prompt = f"Texts: {cropped_collection}"
    if blacklist is not None and len(blacklist) > 0:
        user_prompt += f"\nAvoid the following topics: {blacklist}\n"

    if cache_fn is not None:
        prompt_hash = system_prompt + user_prompt
        cache = shelve.open(cache_fn)
        if prompt_hash in cache:
            cached_result = cache[prompt_hash]
            cache.close()
            return cached_result

    agent = Agent(
        model,
        output_type=HypothesisAboutCollection,
        retries=retries,
        instructions=system_prompt,
    )
    try:
        result = agent.run_sync(user_prompt)
        if cache_fn is not None:
            cache = shelve.open(cache_fn)
            cache[prompt_hash] = result
            cache.close()
    except UnexpectedModelBehavior as e:
        raise e
    
    return result