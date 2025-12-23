from askme.makequestions import api, makequestion
from askme.askquestions.check_entailment import check_entailment_nli
from askme.askquestions import models
from askme.rtp import rtp


def main():
    text_collection = [
        "The cat sat on the mat.",
        "The cat in in the box.",
        "The dog barked loudly.",
        "I like cats",
        "I like dogs",
        "The dog is in the yard.",
    ]
    model = api.build_model()
    nli_model, nli_tokenizer = models.make_nli_model(
        model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    tree, usage = rtp.rtp(
        documents=text_collection,
        model_to_make_questions=model,
        model_to_answer_questions=nli_model,
        tokenizer_to_answer_questions=nli_tokenizer,
        max_depth=2,
        verbose=True,
    )
    print("Token Usage:", usage)
    rtp.print_tree(tree)


if __name__ == "__main__":
    main()
