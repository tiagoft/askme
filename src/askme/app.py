import typer
import json

app = typer.Typer()


@app.command()
def ask_question(phrase: str, model_name: str = 'llama3.1:8b'):
    """
    Prints whether the given PHRASE is a yes/no question.
    """
    pass


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
