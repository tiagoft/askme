from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def make_ollama_model(model_name = 'llama3.1:8b') -> OpenAIChatModel:
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    return model


def make_nli_model(model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7') -> tuple[any, any]:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer