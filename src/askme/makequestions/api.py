import os
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from dotenv import load_dotenv
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel

def make_azure_model(model_name : str = "gpt-4o-mini") -> Model:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    endpoint = os.getenv("ENDPOINT")
    api_version = os.getenv("API_VERSION")
    return OpenAIChatModel(
        model_name=model_name,
        provider=AzureProvider(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        ),
        settings=OpenAIChatModelSettings(
            temperature=0.1,
        )
    )

def make_gemini_model(model_name: str = "gemini-2.5-flash-lite") -> Model:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    return GoogleModel(
        model_name=model_name,
        provider=GoogleProvider(
            api_key=api_key,
        ),
    )

def make_ollama_model(model_name = 'gpt-oss:20b') -> Model:
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    return model

def azure_openai_client():
    load_dotenv()
    api_key = os.getenv("API_KEY")

    # Define the endpoint
    endpoint = os.getenv("ENDPOINT")

    # Create the request
    client = openai.AzureOpenAI(
        # This is the default and can be omitted
        api_key=api_key,
        api_version="2025-01-01-preview",
        azure_endpoint=endpoint,
    )
    return client