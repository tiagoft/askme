import typer
import json
from pprint import pprint

from evalsim.similarities import SimilarityCalculator

app = typer.Typer()


@app.command()
def phrases(
    inputs: list[str],
    use_lexical: bool = True,
    use_semantic: bool = True,
    use_logical: bool = True,
):
    calculator = SimilarityCalculator(
        use_logical=use_logical,
        use_semantic=use_semantic,
        use_lexical=use_lexical,
    )
    """
    Performs zero-shot natural language inference on the provided PREMISE
    against the given HYPOTHESIS.
    """
    result = calculator(inputs)
    pprint(result.model_dump(), indent=2)


@app.command()
def from_table(
    file,
    col_text: str = "text",
    col_category: list[str] | None = None,
    use_lexical: bool = True,
    use_semantic: bool = True,
    use_logical: bool = True,
    output_digits: int = 4,
):
    """Calculate similarities from a table file."""
    import pandas as pd
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    calculator = SimilarityCalculator(
        use_logical=use_logical,
        use_semantic=use_semantic,
        use_lexical=use_lexical,
    )
    df = df.dropna()

    if col_category:
        df_category = df[col_category]
        df['_CATEGORY_'] = df_category.astype(str).agg('_'.join, axis=1)
        col = '_CATEGORY_'
        for category in df[col].unique():
            #print(f"Category: {category}")
            category_texts = df[df[col] == category][col_text].tolist()
            result = calculator(category_texts)
            print(f"{category}, {result.lexical.mean:.{output_digits}f} \\pm {result.lexical.std:.{output_digits}f}, {result.semantic.mean:.{output_digits}f} \\pm {result.semantic.std:.{output_digits}f}, {result.logical.mean:.{output_digits}f} \\pm {result.logical.std:.{output_digits}f}")
    else:
        texts = df[col_text].tolist()
        result = calculator(texts)        
        pprint(result.model_dump(), indent=2)
