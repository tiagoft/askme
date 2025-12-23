from insperdatasets.get_dataset import get_dataset
from askme.askquestions import check_entailment, models
from tqdm import tqdm


def main():
    dataset = get_dataset("arxiv2025")
    dataset_val = dataset['validation']
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=model_name)

    hypothesis = "This text is primarily about computer vision."
    hypothesis_neg = "This text is not about computer vision."

    y = []
    y_pred = []
    for idx in tqdm(range(10)):
        text, label = dataset_val[idx]
        premise = text
        label = "entailment" if label == 'cs.CV' else "contradiction"
        y.append(label)
        (
            is_entailment,
            entailment_logit,
            contradiction_logit,
            P_entailment,
        ) = check_entailment.check_entailment_nli(
            model,
            tokenizer,
            premise,
            hypothesis,
            device='cpu',
        )
        (
            is_entailment_neg,
            entailment_logit_neg,
            contradiction_logit_neg,
            P_entailment_neg,
        ) = check_entailment.check_entailment_nli(
            model,
            tokenizer,
            premise,
            hypothesis_neg,
            device='cpu',
        )
        # print(f"Idx {idx}")
        # print(f"Label: {label}")
        # print(f"Premise: {premise[:50]}...")
        # print(f"Hypothesis: {hypothesis}")
        # print(f"Hypothesis Neg: {hypothesis_neg}")
        # print(f"P_entailment: {P_entailment}")
        # print(f"Entailment Logit: {entailment_logit}")
        # print(f"Contradiction Logit: {contradiction_logit}")
        # print(f"P_entailment Neg: {P_entailment_neg}")
        # print(f"Entailment Logit Neg: {entailment_logit_neg}")
        # print(f"Contradiction Logit Neg: {contradiction_logit_neg}")
        
        
        if P_entailment > P_entailment_neg:
            y_pred.append("entailment")
        else:
            y_pred.append("contradiction")
    
        #print(P_entailment, P_entailment_neg, label)
    from sklearn.metrics import classification_report
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
