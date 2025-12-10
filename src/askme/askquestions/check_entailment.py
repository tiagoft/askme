import torch

def check_entailment_nli(model,
                         tokenizer,
                    premise: str,
                    hypothesis: str,
                    device='cuda:0',
                    label_names = ['contradiction', 'entailment', 'neutral']
                    ) -> tuple[bool, float, float, float]:
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    print(prediction)
    return (prediction["entailment"] > prediction["contradiction"]), prediction["entailment"], prediction["contradiction"]