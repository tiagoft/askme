import torch
import numpy as np
from ..utils import chunk_text




def pool_nli_scores(
    check_fn,
    premise: str,
    hypothesis: str,
    chunk_size: int = 350,
    overlap: int = 50,
    **kwargs
) -> tuple[bool, float, float, float]:
    """
    Apply max-pooling over NLI scores from multiple chunks of a document.
    
    Args:
        check_fn: The NLI checking function to use (e.g., check_entailment_nli)
        premise: The document text (will be chunked if > chunk_size words)
        hypothesis: The hypothesis to check against
        chunk_size: Number of words per chunk (default: 350)
        overlap: Number of words to overlap between chunks (default: 50)
        **kwargs: Additional arguments to pass to check_fn
    
    Returns:
        Tuple of (is_entailed, entailment_score, contradiction_score, P_entailment)
        from the chunk with the highest entailment score (max-pooling)
    """
    chunks = chunk_text(premise, chunk_size, overlap)
    
    results = []
    for chunk in chunks:
        result = check_fn(premise=chunk, hypothesis=hypothesis, **kwargs)
        results.append(result)
    
    # Max-pooling: select the chunk with highest entailment score
    max_entailment_idx = max(range(len(results)), key=lambda i: results[i][1])
    best_result = results[max_entailment_idx]
    
    return best_result 

def check_entailment_nli_pipeline(
    pipeline,
    premise: str,
    hypothesis: str,
    label_names=['contradiction', 'entailment', 'neutral']
) -> tuple[bool, float, float, float]:
    output = pipeline(premise, hypothesis)
    logits = torch.tensor(output['scores'])

    named_logits = {
        name: round(float(logit.item()), 1)
        for logit, name in zip(logits, label_names)
    }

    # P_entailment = P(entailment | entailment or contradiction)
    # We ignore "neutral" for this probability calculation
    # Following: https://arxiv.org/pdf/2303.08896
    # (self-check GPT)
    P_entailment = torch.softmax(
        logits[[
            label_names.index("entailment"),
            label_names.index("contradiction")
        ]], -1)[0].item()
    
    return (
        (named_logits["entailment"] > named_logits["contradiction"]),
        named_logits["entailment"],
        named_logits["contradiction"],
        P_entailment,
    )

def check_entailment_nli(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    device='cuda:0',
    label_names=["entailment", "neutral", "contradiction"]
) -> tuple[bool, float, float, float]:
    tokens = tokenizer(premise,
                       hypothesis,
                       truncation=True,
                       return_tensors="pt")
    output = model(
        tokens["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    logits = output["logits"][0]

    named_logits = {
        name: round(float(logit.item()), 1)
        for logit, name in zip(logits, label_names)
    
    }
    
    # P_entailment = P(entailment | entailment or contradiction)
    # We ignore "neutral" for this probability calculation
    # Following: https://arxiv.org/pdf/2303.08896
    # (self-check GPT)
    P_entailment = np.exp(named_logits["entailment"]) / (
        np.exp(named_logits["entailment"]) + np.exp(named_logits["contradiction"])) 
    
    return (
        (named_logits["entailment"] > named_logits["contradiction"]),
        named_logits["entailment"],
        named_logits["contradiction"],
        P_entailment,
    )
