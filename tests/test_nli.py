from askme.rtp import nli
from askme.config.config import NLIBatchingChukingConfig, config_factory

def test_nli_default_config():
    cfg = config_factory(NLIBatchingChukingConfig)
    assert cfg.model_name == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    assert cfg.chunk_size == 350


def test_nli_operation():
    nli_model = nli.NLIWithChunkingAndPooling()
    premises = ["The sky is blue.", "Grass is green", "It was a dark night"]
    hypothesis = "The sky is clear and blue."
    results = nli_model(premises, hypothesis)
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, nli.NLIResults)