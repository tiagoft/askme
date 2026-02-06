

### Answer questions about a text

#### Basic usage
```python
from askme.rtp.nli import NLIWithChunkingAndPooling
nli_model = nli.NLIWithChunkingAndPooling()
premises = ["The sky is blue.", "Grass is green", "It was a dark night"]
hypothesis = "The sky is clear and blue."
results = nli_model(premises, hypothesis)
print(results)
```



### Make a question about a collection

#### Basic usage
```python
from askme.makequestions.makequestion import QuestionMaker

qm = QuestionMaker()
texts = ['text1', 'text2', 'text3']
result = qm(texts)
print(result.hypothesis)
```

#### Redefining default configurations
```python
from askme.makequestions.makequestion import QuestionMaker
from askme.config.config import config_factory, MakeQuestionsConfig

cfg = config_factory(MakeQuestionsConfig)
cfg.max_words_per_text = 50
qm = QuestionMaker(cfg)
texts = ['text1', 'text2', 'text3']
result = qm(texts)
print(result.hypothesis)
```
