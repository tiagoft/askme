
from .rtp.nli import NLIWithChunkingAndPooling, NLIResults
from .preprocess_pdf import get_document
from pydantic import BaseModel

class DocumentEntailmentResult(BaseModel):
    filename: str
    hypothesis: str
    results: NLIResults

def run_nli_on_single_doc(
    document: str,
    hypothesis: str,
):
    nli_model = NLIWithChunkingAndPooling()
    results = nli_model([document], hypothesis)
    return results[0]

def run_nli_on_files(
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
    

    
    return all_results
    