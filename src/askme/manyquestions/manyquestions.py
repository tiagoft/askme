from pydantic import BaseModel

from askme.config.config import (
    MakeQuestionsConfig,
    NLIBatchingChukingConfig,
    SamplingConfig,
    TextEmbeddingConfig,
    config_factory,
)
from askme.rtp.make_collection_index import make_faiss_index
from askme.utils import TextEmbeddingWithChunker
from askme.utils.sampling import sampler_factory

from .collection_answerer import CollectionAnswerer, QuestionWithAnswers
from .question_generator import ManyQuestionsGenerator


class ManyQuestionsResult(BaseModel):
    """Output of the full ManyQuestions pipeline."""
    questions: list[str]
    question_answers: list[QuestionWithAnswers]
    sampled_indices: list[int]


class ManyQuestions:
    """End-to-end pipeline: sample texts → generate questions → answer across collection.

    Pipeline:
    1. Embed all texts and build a FAISS index.
    2. Sample N representative texts using the configured sampler.
    3. Prompt an LLM with the N sampled texts to produce M yes/no questions.
    4. Run NLI on the full collection for each of the M questions.

    The two sub-steps can also be called independently:
    - Use ``generate()`` to obtain questions and sampled indices without NLI.
    - Use :class:`CollectionAnswerer` directly to answer pre-existing questions.

    Args:
        n_sample: Number of texts to sample for LLM question generation.
        n_questions: Number of yes/no questions for the LLM to produce.
        use_gpu: Whether to use GPU for embedding and FAISS operations.
        embedding_config: Config for the text embedding model.
        llm_config: Config for the LLM question generator.
        nli_config: Config for the NLI answering model.
        sampler_config: Config for the sampling strategy. If None, defaults to
            the project config with ``n_select`` set to ``n_sample``.
    """

    def __init__(
        self,
        n_sample: int = 10,
        n_questions: int = 5,
        use_gpu: bool = False,
        embedding_config: TextEmbeddingConfig | None = None,
        llm_config: MakeQuestionsConfig | None = None,
        nli_config: NLIBatchingChukingConfig | None = None,
        sampler_config: SamplingConfig | None = None,
    ):
        if embedding_config is None:
            embedding_config = config_factory(TextEmbeddingConfig)
        if llm_config is None:
            llm_config = config_factory(MakeQuestionsConfig)
        if nli_config is None:
            nli_config = config_factory(NLIBatchingChukingConfig)
        if sampler_config is None:
            sampler_config = config_factory(SamplingConfig)
            sampler_config.n_select = n_sample
            sampler_config.use_gpu = use_gpu

        self.n_sample = n_sample
        self.n_questions = n_questions
        self.use_gpu = use_gpu
        self.sampler_config = sampler_config
        self.embedding_model = TextEmbeddingWithChunker(config=embedding_config)
        self.question_generator = ManyQuestionsGenerator(
            n_questions=n_questions, config=llm_config
        )
        self.collection_answerer = CollectionAnswerer(config=nli_config)

    def generate(self, collection: list[str]) -> tuple[list[str], list[int]]:
        """Sample N texts and generate M yes/no questions (no NLI answering).

        Useful when you want only the questions, or want to inspect/edit them
        before running :class:`CollectionAnswerer` separately.

        Args:
            collection: Full list of text documents.

        Returns:
            Tuple of ``(questions, sampled_indices)``.
        """
        texts = list(collection)
        dimension = len(self.embedding_model(texts[0]))
        faiss_index, embeddings = make_faiss_index(
            texts,
            embedding_model=self.embedding_model,
            dimension=dimension,
            use_gpu=self.use_gpu,
            return_embeddings=True,
        )
        sampler = sampler_factory(self.sampler_config)
        sampled_indices = sampler(faiss_index=faiss_index, X=embeddings)
        sampled_texts = [texts[i] for i in sampled_indices]
        questions = self.question_generator(sampled_texts)
        return questions, [int(i) for i in sampled_indices]

    def __call__(self, collection: list[str]) -> ManyQuestionsResult:
        """Run the full pipeline: sample → generate → answer.

        Args:
            collection: Full list of text documents.

        Returns:
            :class:`ManyQuestionsResult` with questions, per-document NLI
            answers, and the indices of the texts used for question generation.
        """
        texts = list(collection)
        questions, sampled_indices = self.generate(texts)
        question_answers = self.collection_answerer(texts, questions)
        return ManyQuestionsResult(
            questions=questions,
            question_answers=question_answers,
            sampled_indices=sampled_indices,
        )
