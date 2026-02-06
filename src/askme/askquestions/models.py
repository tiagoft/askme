from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

def make_ollama_model(model_name = 'llama3.1:8b') -> OpenAIChatModel:
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    return model

