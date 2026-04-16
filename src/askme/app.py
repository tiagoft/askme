import typer
import json
from pprint import pprint

app = typer.Typer()

@app.command()
def nlidemo(
    premise: str,
    hypothesis: str,
):
    from .app_nli import run_nli_on_single_doc
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
    from .app_nli import run_nli_on_files
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
def extract_dir_info(
    path_to_dir: str,
    recipe: str,
    output_file : str | None = None,
    n_max : int | None = None,
):
    """
    Extracts information from the provided directory of documents according to the given RECIPE.
    """
    from .app_dataset import app_search_on_directory
    issues = app_search_on_directory(
        path_to_dir,
        config_file=recipe,
        n_max=n_max,
    )
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump([issue.model_dump() for issue in issues], f, indent=2)
    else:
        for issue in issues:
            pprint(issue.model_dump(), indent=2)



@app.command()
def generate_questions(
    input_file: str = typer.Argument(..., help="JSON file containing a list of text strings"),
    n_sample: int = typer.Option(10, help="Number of texts to sample for question generation"),
    n_questions: int = typer.Option(5, help="Number of yes/no questions to generate"),
    model_name: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    use_gpu: bool = typer.Option(False, help="Use GPU for embedding and sampling"),
    output_file: str | None = typer.Option(None, help="Output JSON file (prints to stdout if omitted)"),
):
    """
    Generate M yes/no questions about a text collection by sampling N texts and
    prompting an LLM. The output JSON contains the questions and the indices of
    the sampled texts that were used to generate them.
    """
    from .manyquestions import ManyQuestions
    from .config.config import config_factory, MakeQuestionsConfig

    with open(input_file) as f:
        collection = json.load(f)

    llm_config = config_factory(MakeQuestionsConfig)
    llm_config.model_name = model_name

    pipeline = ManyQuestions(
        n_sample=n_sample,
        n_questions=n_questions,
        use_gpu=use_gpu,
        llm_config=llm_config,
    )
    questions, sampled_indices = pipeline.generate(collection)

    output = {"questions": questions, "sampled_indices": sampled_indices}
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
    else:
        pprint(output, indent=2)


@app.command()
def answer_questions(
    input_file: str = typer.Argument(..., help="JSON file containing a list of text strings"),
    questions_file: str = typer.Argument(
        ...,
        help="JSON file with questions: either a list of strings or "
             '{"questions": [...]} as produced by generate-questions',
    ),
    output_file: str | None = typer.Option(None, help="Output JSON file (prints to stdout if omitted)"),
):
    """
    Answer yes/no questions across a text collection using NLI.
    Produces per-document entailment scores for each question.
    """
    from .manyquestions import CollectionAnswerer

    with open(input_file) as f:
        collection = json.load(f)
    with open(questions_file) as f:
        questions_data = json.load(f)

    if isinstance(questions_data, dict):
        questions = questions_data["questions"]
    else:
        questions = questions_data

    answerer = CollectionAnswerer()
    results = answerer(collection=collection, questions=questions)

    output = [r.model_dump() for r in results]
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
    else:
        pprint(output, indent=2)


@app.command()
def manyquestions(
    input_file: str = typer.Argument(..., help="JSON file containing a list of text strings"),
    n_sample: int = typer.Option(10, help="Number of texts to sample for question generation"),
    n_questions: int = typer.Option(5, help="Number of yes/no questions to generate"),
    model_name: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    use_gpu: bool = typer.Option(False, help="Use GPU for embedding and sampling"),
    output_file: str | None = typer.Option(None, help="Output JSON file (prints to stdout if omitted)"),
):
    """
    End-to-end pipeline: sample N texts, generate M yes/no questions via LLM,
    then answer all questions across the full collection using NLI.
    """
    from .manyquestions import ManyQuestions
    from .config.config import config_factory, MakeQuestionsConfig

    with open(input_file) as f:
        collection = json.load(f)

    llm_config = config_factory(MakeQuestionsConfig)
    llm_config.model_name = model_name

    pipeline = ManyQuestions(
        n_sample=n_sample,
        n_questions=n_questions,
        use_gpu=use_gpu,
        llm_config=llm_config,
    )
    result = pipeline(collection)

    output = result.model_dump()
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
    else:
        pprint(output, indent=2)


@app.command()
def goodbye(name: str = "Friend"):
    """
    Says goodbye to NAME, or "Friend" if no name is given.
    """
    print(f"Goodbye, {name}!")


if __name__ == "__main__":
    app()
