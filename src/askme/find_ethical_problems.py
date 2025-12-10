import toml
import pathlib
from pydantic import BaseModel
import tqdm

from .preprocess_pdf import get_document
from .ask import AskGroundedQuestion
from .askquestions.ask_question import GroundedAnswer


def load_ethical_llm_config():
    config_path = pathlib.Path(
        __file__).parent / "assets" / "ethical_llm_config.toml"
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config


class EthicalProblem(BaseModel):
    problem_name: str | None = None
    depth: str | None = None
    evidence: str | None = None
    page : int | None = None

def get_ethics_problems(
    pdf_path: str | pathlib.Path,
    model_name: str = 'llama3.1:8b',
    start_page: int = 0,
    end_page: int = -1,
) -> list[EthicalProblem]:
    config = load_ethical_llm_config()

    _, document_paginated = get_document(pdf_path)
    document_paginated = document_paginated[start_page:end_page]
    print(f"Document loaded. Number of pages: {len(document_paginated)} (from page {start_page} to {end_page if end_page != -1 else 'end'})")
    model_ask = AskGroundedQuestion(
        model_name=model_name,
        temperature=config['config'].get('temperature', 0.2),
    )

    problems = []
    for problem_name, problem_config in config.get('problems', {}).items():
        print("Looking for ethical issues:", problem_name)
        system_prompt = problem_config['system_prompt']
        for page_num, page_text in tqdm.tqdm(enumerate(document_paginated)):
            response = model_ask(
                instructions=system_prompt,
                material=page_text,
            )
            if response.output.answer:
                this_problem = EthicalProblem(
                    problem_name=problem_name,
                    evidence=response.output.evidence,
                    page=page_num + start_page,
                )
                problems.append(this_problem)
                #print(f"  Found {problem_name} issue on page {page_num + start_page}: {response.output.evidence}")

    print("Ethical problem extraction completed.")
    print("Problems found:", len(problems))

    depth_evaluations = []
    for depth_name, depth_config in config.get('depth', {}).items():
        print("Looking for depth issue:", depth_name)
        system_prompt = depth_config['system_prompt']
        for problem in tqdm.tqdm(problems):
            #print(problem)
            response = model_ask(
                instructions=system_prompt,
                material=problem.problem_name + " - " + problem.evidence,
            )
            if response.output.answer:
                this_depth = EthicalProblem(
                    problem_name=problem.problem_name,
                    depth=depth_name,
                    evidence=response.output.evidence,
                    page=problem.page,
                )
                depth_evaluations.append(this_depth)
                #print(f"  Found {depth_name} depth on problem {problem.problem_name} on page {problem.page}: {response.output.evidence}")
    return depth_evaluations
