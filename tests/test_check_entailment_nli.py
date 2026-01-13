from askme.askquestions import check_entailment, models
from askme.utils import NLIWithChunkingAndPooling
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


def test_nli_with_chunking_and_pooling():
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=model_name)
    model = model.to('cpu')
    nli_model = NLIWithChunkingAndPooling(
        nli_model = model,
        tokenizer = tokenizer,
        chunk_size=50,
        overlap=10,
        device='cpu'
    )
    
    premise = ["This is a long premise text. " * 100,
                "Another long premise text goes here." * 100]
    hypothesis = "This is a hypothesis."
    
    results = nli_model(
        premise=premise,
        hypothesis=hypothesis,
    )
    
    assert len(results) == len(premise)
    for res in results:
        assert isinstance(res, tuple)
        assert len(res) == 4