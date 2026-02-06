import typer
import json
from pprint import pprint
from .app_nli import run_nli_on_files, run_nli_on_single_doc
app = typer.Typer()

@app.command()
def nlidemo(
    premise: str,
    hypothesis: str,
):
    """
    Performs zero-shot natural language inference on the provided PREMISE
    against the given HYPOTHESIS.
    """
    result = run_nli_on_single_doc(premise, hypothesis)
    pprint(result.model_dump(), indent=2)

@app.command()
def basenli(
    documents: list[str],
    hypothesis: list[str] = None,
    output_file: str | None = None,
):
    """
    Performs zero-shot natural language inference on the provided DOCUMENTS
    against the given HYPOTHESIS.
    """
    
    if hypothesis is None:
        hypothesis = "The document is relevant to the question."
    elif isinstance(hypothesis, list) and len(hypothesis) == 1:
        hypothesis = hypothesis[0]
    results = run_nli_on_files(documents, hypothesis)
    if output_file is None:
        for i, result in enumerate(results):
            pprint(result.model_dump(), indent=2)       
    else:
        with open(output_file, 'w') as f:
            json.dump([result.model_dump() for result in results], f, indent=2)        


@app.command()
def extract_ethical_issues(
    path_to_pdf: str,
    model_name: str = 'llama3.1:8b',
    start_page: int = 0,
    end_page: int = -1,
    output_file : str | None = None,
):
    """
    Extracts ethical issues from the provided MATERIAL.
    """
    from .find_ethical_problems import get_ethics_problems
    issues = get_ethics_problems(
        path_to_pdf,
        model_name=model_name,
        start_page=start_page,
        end_page=end_page,
    )
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump([issue.model_dump() for issue in issues], f, indent=2)
    


@app.command()
def goodbye(name: str = "Friend"):
    """
    Says goodbye to NAME, or "Friend" if no name is given.
    """
    print(f"Goodbye, {name}!")


if __name__ == "__main__":
    app()
