from askme.config.config import config_factory, MakeQuestionsConfig, SamplingConfig

def test_config_factory_default():
    cfg = config_factory(MakeQuestionsConfig)
    assert cfg.model_name == "gpt-oss:20b"
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 150
    assert cfg.top_p == 1.0
    assert cfg.frequency_penalty == 0.0
    assert cfg.presence_penalty == 0.0
    assert cfg.max_words_per_text == 350
    assert cfg.retries == 10
    assert cfg.blacklist == []
    
def test_config_factory_override():
    cfg = config_factory(SamplingConfig, override_data="NLISamplingConfig")
    assert cfg.n_select == 0.1