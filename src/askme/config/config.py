"""Configuration models for the AskMe applications."""

from pydantic import BaseModel
from typing import Optional
import toml
from askme.assets import config_data as default_config_data


def config_factory(
    config_class: type[BaseModel],
    config_data: dict | None = None,
) -> BaseModel:
    if config_data is None:
        config_data = default_config_data[config_class.__name__]
    
    return config_class(**config_data)


class MakeQuestionsConfig(BaseModel):
    """Configuration for making questions."""

    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_words_per_text: int = 350
    retries: int = 10
    blacklist: Optional[list[str]] = []
