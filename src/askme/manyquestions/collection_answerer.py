from pydantic import BaseModel
from tqdm import tqdm

from askme.config.config import NLIBatchingChukingConfig, config_factory
from askme.rtp.nli import NLIResults, NLIWithChunkingAndPooling


class DocumentQuestionAnswer(BaseModel):
    """NLI answer for a single (document, question) pair."""
    document_index: int
    is_entailed: bool
    entailment_score: float
    contradiction_score: float
    P_entailment_binary: float


class QuestionWithAnswers(BaseModel):
    """A yes/no question paired with NLI answers across a document collection."""
    question: str
    answers: list[DocumentQuestionAnswer]


class CollectionAnswerer:
    """Answers yes/no questions across a text collection using NLI.

    For each question, runs NLI with the full collection as premises and the
    question as hypothesis, returning per-document entailment scores.

    Args:
        config: NLI configuration. Defaults to NLIBatchingChukingConfig from config.toml.
    """

    def __init__(self, config: NLIBatchingChukingConfig | None = None):
        if config is None:
            config = config_factory(NLIBatchingChukingConfig)
        self.nli = NLIWithChunkingAndPooling(config=config)

    def __call__(
        self,
        collection: list[str],
        questions: list[str],
    ) -> list[QuestionWithAnswers]:
        """Answer each question for every document in the collection.

        Args:
            collection: Full list of text documents.
            questions: List of yes/no questions to evaluate.

        Returns:
            One QuestionWithAnswers per question, each containing per-document
            NLI results.
        """
        results = []
        for question in tqdm(questions, desc="Answering questions"):
            nli_results: list[NLIResults] = self.nli(
                premise=collection, hypothesis=question
            )
            answers = [
                DocumentQuestionAnswer(
                    document_index=i,
                    is_entailed=r.is_entailed,
                    entailment_score=r.entailment_score,
                    contradiction_score=r.contradiction_score,
                    P_entailment_binary=r.P_entailment_binary,
                )
                for i, r in enumerate(nli_results)
            ]
            results.append(QuestionWithAnswers(question=question, answers=answers))
        return results
