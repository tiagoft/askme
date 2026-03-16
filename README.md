# ASKME

Using AI/NLP to create and ask questions about texts.

## Installing

### As user (use as a library)

```bash
pip install git+https://github.com/tiagoft/askme.git
```

### As developer (to contribute/develop)

#### Step 1: Install `uv`

Follow instructions on https://github.com/astral-sh/uv.

#### Step 2: Clone and sync repo

```bash
git clone git@github.com:tiagoft/askme.git
cd askme
uv sync
source .venv/bin/activate
```

To add new libraries:

```bash
uv add name_of_pip_package
```


## Use cases and code examples

### Answer questions about a text

#### Using NLI

As a Python library:

```python
from askme.rtp.nli import NLIWithChunkingAndPooling
nli_model = nli.NLIWithChunkingAndPooling()
premises = ["The sky is blue.", "Grass is green", "It was a dark night"]
hypothesis = "The sky is clear and blue."
results = nli_model(premises, hypothesis)
print(results)
```

From command line:

```bash
askme nlidemo "The sky is blue" "The sky is clear and blue"
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

### Evaluation of similarities

Check [evalsim documentation](src/evalsim/README.md).
