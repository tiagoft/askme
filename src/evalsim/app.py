import typer
import json
from pprint import pprint

from evalsim.similarities import SimilarityCalculator

app = typer.Typer()



@app.command()
def evalsim(
    inputs: list[str],
    use_lexical : bool = True,
    use_semantic: bool  = True,
    use_logical: bool = True,
    
):
    calculator = SimilarityCalculator(
        use_logical=use_logical,
        use_semantic=use_semantic,
        use_lexical=use_lexical,)
    """
    Performs zero-shot natural language inference on the provided PREMISE
    against the given HYPOTHESIS.
    """
    result = calculator(inputs)
    pprint(result.model_dump(), indent=2)
    