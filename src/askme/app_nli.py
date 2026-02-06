
from .rtp.nli import NLIWithChunkingAndPooling
from .preprocess_pdf import get_document

def run_nli(
    text_paths: list[str],
    hypothesis: str,
):
    nli_model = NLIWithChunkingAndPooling()
    documents = []
    for path in text_paths:
        if path.endswith('.pdf'):
            document, _ = get_document(path)
        else:
            with open(path, 'r') as f:
                document = f.read()
        documents.append(document)
    
    results = nli_model(documents, hypothesis)
    return results
    
    