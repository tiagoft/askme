from insperdatasets.get_dataset import get_dataset
from askme.askquestions import check_entailment, models
from tqdm import tqdm


def main():
    dataset = get_dataset("arxiv2025")
    dataset_val = dataset['validation']
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    pipe = models.make_nli_pipeline(model_name=model_name)

    hypothesis = "This paper is primarily about computer vision."

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
        ) = check_entailment.check_entailment_nli_pipeline(
            pipe,
            premise,
            hypothesis,
        )
        if is_entailment:
            y_pred.append("entailment")
        else:
            y_pred.append("contradiction")
        
    from sklearn.metrics import classification_report
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
