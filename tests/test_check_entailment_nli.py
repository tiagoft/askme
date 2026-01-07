from askme.askquestions import check_entailment, models
import pytest
import transformers

#     check_entailment_nli(
#     model,
#     tokenizer,
#     premise: str,
#     hypothesis: str,
#     device='cuda:0',
#     label_names=['contradiction', 'entailment', 'neutral']
# )

def test_check_entailment_nli():
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=model_name)
    check_entailment.check_entailment_nli(
        model,
        tokenizer,
        premise="The sky is blue.",
        hypothesis="The sky is clear.",
        device='cpu'
    )

@pytest.mark.gpu
def test_check_entailment_pipeline():
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=model_name)
    model = model.to('cuda')
    check_entailment.check_entailment_nli(
        model,
        tokenizer,
        premise="The sky is blue.",
        hypothesis="The sky is clear.",
    )