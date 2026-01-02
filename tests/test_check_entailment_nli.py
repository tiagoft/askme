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
    
def test_check_entailment_pipeline():
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    pipe = models.make_nli_pipeline(model_name=model_name, use_cuda=False)
    check_entailment.check_entailment_nli_pipeline(
        pipeline = pipe,
        premise="The sky is blue.",
        hypothesis="The sky is clear.",
    )