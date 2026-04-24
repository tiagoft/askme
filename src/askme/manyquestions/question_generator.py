import logging
import shelve
from pathlib import Path

from typing import Annotated
from pydantic import Field, create_model
from pydantic_ai import Agent, UnexpectedModelBehavior

logger = logging.getLogger(__name__)

from askme.assets import rtp_prompts
from askme.config.config import MakeQuestionsConfig, config_factory
from askme.makequestions.api import make_azure_model, make_ollama_model
from askme.makequestions.makequestion import crop_text_in_words



def _questions_model(n: int):
    return create_model(
        "QuestionsAboutCollection",
        questions=(Annotated[list[str], Field(min_length=2, max_length=2 * n)], ...),
    )


class ManyQuestionsGenerator:
    """Generates multiple yes/no questions about a text collection using an LLM.

    The generator takes a (pre-sampled) list of texts and produces N distinct
    yes/no questions that characterize the collection's themes and content.

    Args:
        n_questions: Number of yes/no questions to generate.
        config: LLM configuration. Defaults to MakeQuestionsConfig from config.toml.
    """

    def __init__(
        self,
        n_questions: int = 5,
        config: MakeQuestionsConfig | None = None,
    ):
        if config is None:
            config = config_factory(MakeQuestionsConfig)
        self.config = config
        self.n_questions = n_questions

        if self.config.model_name.startswith('gpt-4o'):
            self.llm_model = make_azure_model(self.config.model_name)
        else:
            self.llm_model = make_ollama_model(self.config.model_name)
        self.cache_fn = Path(self.config.cache).expanduser()

    def __call__(self, collection: list[str]) -> list[str]:
        """Generate yes/no questions about the collection.

        Args:
            collection: Pre-sampled list of text strings to generate questions from.

        Returns:
            List of yes/no question strings (length == self.n_questions).
        """
        return make_questions_about_collection(
            collection,
            n_questions=self.n_questions,
            model=self.llm_model,
            model_name=self.config.model_name,
            retries=self.config.retries,
            max_words_per_text=self.config.max_words_per_text,
            cache_fn=self.cache_fn,
        )


def make_questions_about_collection(
    collection: list[str],
    n_questions: int,
    model_name: str,
    model,
    retries: int = 10,
    max_words_per_text: int = 350,
    cache_fn=None,
) -> list[str]:
    """Generate N yes/no questions about a collection of texts.

    Args:
        collection: List of text strings (pre-sampled representative subset).
        n_questions: Number of distinct yes/no questions to generate.
        model: Pydantic-AI model instance.
        retries: Number of LLM retries on validation failure.
        max_words_per_text: Maximum words per text before cropping.
        cache_fn: Path to shelve cache file (skips LLM call on cache hit).

    Returns:
        List of yes/no question strings.
    """
    system_prompt = rtp_prompts['manyquestions']['system_prompt'].format(num_questions=n_questions)
    cropped = [crop_text_in_words(t, max_words_per_text) for t in collection]
    user_prompt = f"\n\nTexts: {cropped}"

    if cache_fn is not None:
        prompt_hash = system_prompt + user_prompt + model_name
        cache = shelve.open(str(cache_fn))
        if prompt_hash in cache:
            cached = cache[prompt_hash]
            cache.close()
            return cached
        cache.close()

    agent = Agent(
        model,
        output_type=_questions_model(n_questions),
        retries=retries,
        instructions=system_prompt,
    )
    try:
        result = agent.run_sync(user_prompt)
        questions = result.output.questions
        if cache_fn is not None:
            cache = shelve.open(str(cache_fn))
            cache[prompt_hash] = questions
            cache.close()
    except UnexpectedModelBehavior as e:
        raise e

    logger.info(
        "Generated %d questions (model=%s):\n%s",
        len(questions),
        model_name,
        "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(questions)),
    )
    return questions
