import torch
import numpy as np 

def check_entailment_nli_pipeline(
    pipeline,
    premise: str,
    hypothesis: str,
    label_names=['contradiction', 'entailment', 'neutral']
) -> tuple[bool, float, float, float]:
    output = pipeline(
        {
            'text': premise,
            'hypothesis': hypothesis,
        }
    )[0]
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
