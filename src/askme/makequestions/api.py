import os
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from dotenv import load_dotenv
from pydantic_ai.providers.azure import AzureProvider


def build_model(model_name : str = "gpt-4o-mini") -> Model:
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