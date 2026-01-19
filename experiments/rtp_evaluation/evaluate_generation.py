import askme.makequestions.api as api
import time
from pydantic import BaseModel
from pydantic_ai import Agent

class HelloWorldResponse(BaseModel):
    message: str


def generate():
    model = api.make_ollama_model('qwen3:14b')
    prompt = "Respond with 'Hello, World!'"
    agent = Agent(
        model=model,
        output_type=HelloWorldResponse,
        instructions=prompt,
    )
    result = agent.run_sync("")
    return result.output.message


def run():
    
    
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate generation performance")
    parser.add_argument("--input_model",
                        type=str,
                        required=True,
                        help="Name of the model to use for generation",)
    
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    for _ in range(3):
        t0 = time.time()
        s = generate()
        print(s)
        print("Time taken:", time.time() - t0)
        