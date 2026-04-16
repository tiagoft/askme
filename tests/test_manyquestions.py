"""Tests for the manyquestions module.

Default run (pytest with no flags) covers:
  - Pydantic model correctness
  - ManyQuestionsGenerator initialisation
  - CollectionAnswerer (NLI only, no API)

Tests marked @pytest.mark.llm require a live LLM API and are excluded by
default (see pyproject.toml addopts).
"""

import pytest

from askme.manyquestions import (
    CollectionAnswerer,
    DocumentQuestionAnswer,
    ManyQuestions,
    ManyQuestionsGenerator,
    ManyQuestionsResult,
    QuestionsAboutCollection,
    QuestionWithAnswers,
)
from askme.config.config import config_factory, MakeQuestionsConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

sample_collection = [
    "The cat sat on the mat.",
    "The cat is in the box.",
    "The dog barked loudly.",
    "I like cats.",
    "I like dogs.",
    "The dog is in the yard.",
    "Birds can fly high in the sky.",
    "Fish swim in the ocean.",
    "Elephants are the largest land animals.",
    "Lions are known as the kings of the jungle.",
]

# ---------------------------------------------------------------------------
# Pydantic model tests (no model loading)
# ---------------------------------------------------------------------------

def test_questions_about_collection_model():
    q = QuestionsAboutCollection(questions=["Is the sky blue?", "Is water wet?"])
    assert len(q.questions) == 2
    assert q.questions[0] == "Is the sky blue?"


def test_document_question_answer_model():
    dqa = DocumentQuestionAnswer(
        document_index=2,
        is_entailed=True,
        entailment_score=0.85,
        contradiction_score=0.05,
        P_entailment_binary=0.85,
    )
    assert dqa.document_index == 2
    assert dqa.is_entailed is True
    assert dqa.entailment_score == pytest.approx(0.85)


def test_question_with_answers_model():
    dqa = DocumentQuestionAnswer(
        document_index=0,
        is_entailed=True,
        entailment_score=0.8,
        contradiction_score=0.1,
        P_entailment_binary=0.8,
    )
    qwa = QuestionWithAnswers(question="The text focuses on animals.", answers=[dqa])
    assert qwa.question == "The text focuses on animals."
    assert len(qwa.answers) == 1
    assert qwa.answers[0].document_index == 0


def test_many_questions_result_model():
    dqa = DocumentQuestionAnswer(
        document_index=0,
        is_entailed=False,
        entailment_score=0.2,
        contradiction_score=0.7,
        P_entailment_binary=0.2,
    )
    qwa = QuestionWithAnswers(question="The text focuses on science.", answers=[dqa])
    result = ManyQuestionsResult(
        questions=["The text focuses on science."],
        question_answers=[qwa],
        sampled_indices=[3, 7],
    )
    assert len(result.questions) == 1
    assert result.sampled_indices == [3, 7]


def test_many_questions_result_model_dump():
    dqa = DocumentQuestionAnswer(
        document_index=0,
        is_entailed=True,
        entailment_score=0.9,
        contradiction_score=0.05,
        P_entailment_binary=0.9,
    )
    qwa = QuestionWithAnswers(question="The text focuses on nature.", answers=[dqa])
    result = ManyQuestionsResult(
        questions=["The text focuses on nature."],
        question_answers=[qwa],
        sampled_indices=[1],
    )
    d = result.model_dump()
    assert "questions" in d
    assert "question_answers" in d
    assert "sampled_indices" in d

# ---------------------------------------------------------------------------
# ManyQuestionsGenerator initialisation (no LLM call)
# ---------------------------------------------------------------------------

def test_many_questions_generator_stores_n_questions():
    gen = ManyQuestionsGenerator(n_questions=4)
    assert gen.n_questions == 4


def test_many_questions_generator_accepts_custom_config():
    cfg = config_factory(MakeQuestionsConfig)
    gen = ManyQuestionsGenerator(n_questions=7, config=cfg)
    assert gen.n_questions == 7
    assert gen.config is cfg


# ---------------------------------------------------------------------------
# CollectionAnswerer – NLI only, no LLM API required
# ---------------------------------------------------------------------------

def test_collection_answerer_initialises():
    answerer = CollectionAnswerer()
    assert answerer.nli is not None


def test_collection_answerer_one_result_per_question():
    answerer = CollectionAnswerer()
    questions = ["The text describes an animal.", "The text discusses the sky."]
    results = answerer(collection=sample_collection, questions=questions)
    assert len(results) == len(questions)


def test_collection_answerer_one_answer_per_document():
    answerer = CollectionAnswerer()
    results = answerer(
        collection=sample_collection,
        questions=["The text describes an animal."],
    )
    assert len(results[0].answers) == len(sample_collection)


def test_collection_answerer_output_types():
    answerer = CollectionAnswerer()
    results = answerer(
        collection=["The cat is an animal.", "The sky is blue."],
        questions=["The text focuses on animals."],
    )
    assert isinstance(results, list)
    assert isinstance(results[0], QuestionWithAnswers)
    assert isinstance(results[0].answers[0], DocumentQuestionAnswer)


def test_collection_answerer_scores_in_range():
    answerer = CollectionAnswerer()
    results = answerer(
        collection=["The cat is a furry animal.", "The stock market crashed."],
        questions=["The text focuses on animals."],
    )
    for answer in results[0].answers:
        assert 0.0 <= answer.P_entailment_binary <= 1.0
        assert 0.0 <= answer.entailment_score <= 1.0
        assert 0.0 <= answer.contradiction_score <= 1.0


def test_collection_answerer_relevant_doc_scores_higher():
    """A clearly on-topic document should entail the hypothesis more than an
    unrelated one."""
    answerer = CollectionAnswerer()
    results = answerer(
        collection=[
            "Cats and dogs are common household pets.",
            "The central bank raised interest rates by 2%.",
        ],
        questions=["The text focuses on animals."],
    )
    animal_score = results[0].answers[0].P_entailment_binary
    finance_score = results[0].answers[1].P_entailment_binary
    assert animal_score > finance_score


def test_collection_answerer_question_preserved_in_output():
    answerer = CollectionAnswerer()
    question = "The text focuses on animals."
    results = answerer(collection=sample_collection[:3], questions=[question])
    assert results[0].question == question


def test_collection_answerer_document_indices_are_sequential():
    answerer = CollectionAnswerer()
    collection = sample_collection[:4]
    results = answerer(collection=collection, questions=["The text focuses on animals."])
    indices = [a.document_index for a in results[0].answers]
    assert indices == list(range(len(collection)))


# ---------------------------------------------------------------------------
# LLM-dependent tests
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_many_questions_generator_returns_correct_count():
    gen = ManyQuestionsGenerator(n_questions=3)
    questions = gen(sample_collection[:5])
    assert len(questions) == 3


@pytest.mark.llm
def test_many_questions_generator_returns_nonempty_strings():
    gen = ManyQuestionsGenerator(n_questions=2)
    questions = gen(sample_collection[:5])
    assert all(isinstance(q, str) and len(q) > 0 for q in questions)


@pytest.mark.llm
def test_many_questions_generate_returns_right_counts():
    pipeline = ManyQuestions(n_sample=3, n_questions=2, use_gpu=False)
    questions, sampled_indices = pipeline.generate(sample_collection)
    assert len(questions) == 2
    assert len(sampled_indices) == 3


@pytest.mark.llm
def test_many_questions_generate_indices_within_bounds():
    pipeline = ManyQuestions(n_sample=4, n_questions=2, use_gpu=False)
    _, sampled_indices = pipeline.generate(sample_collection)
    assert all(0 <= i < len(sample_collection) for i in sampled_indices)


@pytest.mark.llm
def test_many_questions_generate_indices_unique():
    pipeline = ManyQuestions(n_sample=4, n_questions=2, use_gpu=False)
    _, sampled_indices = pipeline.generate(sample_collection)
    assert len(set(sampled_indices)) == len(sampled_indices)


@pytest.mark.llm
def test_many_questions_full_pipeline_result_type():
    pipeline = ManyQuestions(n_sample=3, n_questions=2, use_gpu=False)
    result = pipeline(sample_collection)
    assert isinstance(result, ManyQuestionsResult)


@pytest.mark.llm
def test_many_questions_full_pipeline_question_count():
    pipeline = ManyQuestions(n_sample=3, n_questions=2, use_gpu=False)
    result = pipeline(sample_collection)
    assert len(result.questions) == 2
    assert len(result.question_answers) == 2


@pytest.mark.llm
def test_many_questions_full_pipeline_answer_coverage():
    """Every document in the collection must be answered for every question."""
    pipeline = ManyQuestions(n_sample=3, n_questions=2, use_gpu=False)
    result = pipeline(sample_collection)
    for qa in result.question_answers:
        assert len(qa.answers) == len(sample_collection)


# ---------------------------------------------------------------------------
# End-to-end demonstration on a toy two-topic database
# ---------------------------------------------------------------------------

# A small collection with two clearly distinct topics (cooking vs. astronomy).
# The split is deliberate so we can assert that at least one generated question
# discriminates between the groups.
toy_collection = [
    # cooking (indices 0-4)
    "To make pasta, boil salted water and cook the noodles until al dente.",
    "A good risotto requires constant stirring and gradual addition of warm stock.",
    "Bread dough needs time to rise so the yeast can produce carbon dioxide.",
    "Caramelising onions slowly over low heat brings out their natural sweetness.",
    "Tempering chocolate means carefully controlling its temperature while melting.",
    # astronomy (indices 5-9)
    "The Milky Way is a barred spiral galaxy containing over 200 billion stars.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "NASA's James Webb Space Telescope observes the universe in infrared light.",
    "A supernova marks the explosive death of a massive star at the end of its life.",
    "The distance from Earth to the Moon is roughly 384,000 kilometres.",
]


@pytest.mark.llm
@pytest.mark.integration
def test_toy_database_generate_and_answer_question_count():
    """Full pipeline on toy collection returns the requested number of questions
    and one answer per document for each question."""
    pipeline = ManyQuestions(n_sample=6, n_questions=4, use_gpu=False)
    result = pipeline(toy_collection)

    assert len(result.questions) == 4
    assert len(result.question_answers) == 4
    for qa in result.question_answers:
        assert len(qa.answers) == len(toy_collection)


@pytest.mark.llm
@pytest.mark.integration
def test_toy_database_scores_in_range():
    """All NLI probability scores must be valid probabilities."""
    pipeline = ManyQuestions(n_sample=6, n_questions=4, use_gpu=False)
    result = pipeline(toy_collection)

    for qa in result.question_answers:
        for answer in qa.answers:
            assert 0.0 <= answer.P_entailment_binary <= 1.0
            assert 0.0 <= answer.entailment_score <= 1.0
            assert 0.0 <= answer.contradiction_score <= 1.0


@pytest.mark.llm
@pytest.mark.integration
def test_toy_database_cooking_question_scores_higher_on_cooking_docs():
    """A question that focuses on cooking/food should score higher on the
    cooking half of the collection than on the astronomy half, on average."""
    pipeline = ManyQuestions(n_sample=6, n_questions=6, use_gpu=False)
    result = pipeline(toy_collection)

    # Find a question that is clearly about cooking/food.
    cooking_keywords = {"cook", "food", "recipe", "ingredient", "culinary", "dish",
                        "eating", "kitchen", "meal", "preparation"}
    cooking_qa = next(
        (qa for qa in result.question_answers
         if any(kw in qa.question.lower() for kw in cooking_keywords)),
        None,
    )
    if cooking_qa is None:
        pytest.skip("LLM did not generate a cooking-related question in this run")

    cooking_scores = [cooking_qa.answers[i].P_entailment_binary for i in range(5)]
    astronomy_scores = [cooking_qa.answers[i].P_entailment_binary for i in range(5, 10)]

    assert sum(cooking_scores) / len(cooking_scores) > sum(astronomy_scores) / len(astronomy_scores)


@pytest.mark.llm
@pytest.mark.integration
def test_toy_database_generate_then_answer_separately():
    """generate() followed by CollectionAnswerer gives the same structure as
    calling the full pipeline directly."""
    from askme.manyquestions import CollectionAnswerer

    pipeline = ManyQuestions(n_sample=5, n_questions=3, use_gpu=False)
    questions, sampled_indices = pipeline.generate(toy_collection)

    assert len(questions) == 3
    assert len(sampled_indices) == 5
    assert all(0 <= i < len(toy_collection) for i in sampled_indices)

    answerer = CollectionAnswerer()
    question_answers = answerer(collection=toy_collection, questions=questions)

    assert len(question_answers) == 3
    for qa in question_answers:
        assert qa.question in questions
        assert len(qa.answers) == len(toy_collection)
