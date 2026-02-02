

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
