from .askquestions.ask_question import ask_question_ollama, GroundedAnswer
from .askquestions.models import make_ollama_model
from typing import Any
import torch
from functools import partial

class AskGroundedQuestion:
    def __init__(self,
                 model_name: str = 'llama3.1:8b',
                 **model_kwargs):
        self.model = make_ollama_model(model_name=model_name)
        self.model_kwargs = model_kwargs

    def __call__(
        self,
        instructions: str,
        material: str,
    ) -> Any:
        result = ask_question_ollama(
            model=self.model,
            instructions=instructions,
            material=material,
            output_type=GroundedAnswer,
        )
        return result


class CheckEntailment:
    def __init__(self,
                 model_name: str = 'llama3.1:8b',
                 engine='ollama',
                 **model_kwargs):
        self.engine = engine

        if engine == 'ollama':
            self.model = make_ollama_model(model_name=model_name)
            self.entailment_fn

        elif engine == 'transformers':
            self.model, self.tokenizer = make_nli_model(model_name=model_name)
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            if model_name.startswith('cross-encoder'):
                label_names = ['contradiction', 'entailment', 'neutral']
            elif model_name.startswith('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'):
                label_names = ["entailment", "neutral", "contradiction"]
            else:
                label_names = ['contradiction', 'entailment', 'neutral']
                
                
            self.entailment_fn = partial(check_entailment_nli,
                                         model=self.model,
                                         tokenizer=self.tokenizer,
                                         device=self.device,
                                         label_names=label_names)
        else:
            raise ValueError(f'Unsupported engine: {engine}')
        self.model_kwargs = model_kwargs

    def __call__(
        self,
        premise: str,
        hypothesis: str,
    ) -> bool:
        if hasattr(self, 'entailment_fn'):
            result = self.entailment_fn(premise=premise, hypothesis=hypothesis)
            return result[0]
        else:
            raise ValueError('Entailment function not defined for this engine.')

def check_if_is_question(
    phrase: str,
    model_name: str = 'llama3.1:8b',
    output_usage=True,
) -> bool | tuple[bool, Any] | None:
    instructions = ("""Determine if the following phrase is a yes/no question.
                    A yes/no question can be answered with either 'yes' or 'no'.
                    Example: "Do birds fly?" -> yes
                    Example: "Is the sky blue at night?" -> yes
                    Example: "Is this phrase a lie?" -> yes
                    Example: "Tell me about the weather." -> no
                    Example: "What is your name?" -> no
                    Example: "Is this text about physics or math?" -> no
                    """)
    try:
        answer = ask_question(
            model=make_ollama_model(model_name),
            instructions=instructions,
            material=phrase,
        )
    except Exception as e:
        raise e

    if output_usage:
        return answer.output.answer, answer.usage()
    else:
        return answer.output.answer
