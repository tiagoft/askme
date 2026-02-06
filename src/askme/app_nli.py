
from .rtp.nli import NLIWithChunkingAndPooling, NLIResults
from .preprocess_pdf import get_document
from pydantic import BaseModel

class DocumentEntailmentResult(BaseModel):
    filename: str
    hypothesis: str
    results: NLIResults
    
def run_nli(
    text_paths: list[str],
    hypotheses: list[str] | str,
):
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    elif len(hypotheses) != len(text_paths):
        raise ValueError("Length of hypotheses must match length of text_paths.")
    
    nli_model = NLIWithChunkingAndPooling()
    documents = []
    for path in text_paths:
        if path.endswith('.pdf'):
            document, _ = get_document(path)
        else:
            with open(path, 'r') as f:
                document = f.read()
        documents.append(document)
    
    all_results = []
    for hypothesis in hypotheses:
        print(f"Running NLI for hypothesis: {hypothesis}")
        results = nli_model(documents, hypothesis)
        for doc_path, result in zip(text_paths, results):
            all_results.append(DocumentEntailmentResult(
                filename=doc_path,
                hypothesis=hypothesis,
                results=result,
            ))
    
    return all_results
    