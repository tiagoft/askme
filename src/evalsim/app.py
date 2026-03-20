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
def from_excel(
    file,
    col_text: str = "text",
    col_category: str | None = None,
    use_lexical: bool = True,
    use_semantic: bool = True,
    use_logical: bool = True,
):
    """Calculate similarities from an Excel file."""
    import pandas as pd
    df = pd.read_excel(file)
    calculator = SimilarityCalculator(
        use_logical=use_logical,
        use_semantic=use_semantic,
        use_lexical=use_lexical,
    )
    
    results = []
    if col_category:
        df_grouped = df.groupby(col_category)
        for category, group in df_grouped:
            texts = group[col_text].dropna().tolist()
            similarity = calculator(texts)
            results.append({
                "category": category,
                "similarity": similarity.model_dump(),
            })
    else:
        for _, row in df.iterrows():
            texts = [str(row[col]) for col in df.columns if pd.notna(row[col])]
            similarity = calculator(texts)
            results.append(similarity.model_dump())

    pprint(results, indent=2)
