import toml
import pathlib

import tqdm

from .preprocess_pdf import get_document
from .ask import AskGroundedQuestion
from .askquestions.ask_question import GroundedAnswer

def load_ethical_llm_config():
    config_path = pathlib.Path(__file__).parent / "assets" / "ethical_llm_config.toml"
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config

def get_ethics_problems(pdf_path: str | pathlib.Path, model_name: str = 'llama3.1:8b'):
    config = load_ethical_llm_config()
    _, document_paginated = get_document(pdf_path)
    model_ask = AskGroundedQuestion(
        model_name=model_name,
        temperature=config['config'].get('temperature', 0.2),
    )
    
    problems = []
    for problem_name, problem_config in config.get('problems', {}).items():
        print("Looking for problem:", problem_name)
        system_prompt = problem_config['system_prompt']
        for page_num, page_text in tqdm.tqdm(enumerate(document_paginated)):
            response = model_ask(
                instructions=system_prompt,
                material=page_text,
            )
            if response.output.answer:
                problems.append(f"[Page {page_num + 1}] {problem_name.capitalize()} issue found. Evidence: {response.output.evidence}")
            
    return problems