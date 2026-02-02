import toml
from pathlib import Path
import json
from askme.rtp import TreeNode, SplitMetrics
import matplotlib.pyplot as plt
import askme.makequestions.api as api
from pydantic import BaseModel
from pydantic_ai import Agent
import pandas as pd
# Hurray LLM as a judge


class Evaluation(BaseModel):
    q1: bool
    q2: bool
    q3: bool
    q4: bool
    q5: bool
    q6: bool
    q7: bool

class AverageEvaluation(BaseModel):
    q1: float
    q2: float
    q3: float
    q4: float
    q5: float
    q6: float
    q7: float

def evaluate_question(
    role: str,
    question: str,
    labels: list[str],
) -> Evaluation:
    model = api.make_gemini_model(model_name="gemini-2.5-flash")
    #model = api.make_azure_model()
    prompt = f"""You are a {role}. The following predicate was used to divide a set of documents into two groups - those for which the predicate is true, and those for which it is false.
The predicate is: {question}.
The documents were labeled within the following categories: {', '.join(labels)}.

You must judge the following aspects of the predicate. For each question, answer Yes/No (or the specified format) followed by 1–2 sentences of reasoning. Base your judgment strictly on the predicate text, the provided category list, and any grounding examples.

q1. Is this predicate largely redundant with simply assigning documents to one of these categories? In other words, does it mostly replicate at least one of the gold topic boundaries (e.g., it would assign almost the same documents as one of the categories?
   - Yes/No
   - Low-redundancy example: "The text focuses on factual reporting rather than personal opinion" (cuts across all categories).
   - High-redundancy example: "The text is about international politics or conflicts" (almost identical to World category).

q2. Does this predicate apply meaningfully (i.e., creates a non-trivial split of documents) across more than one of these categories?
   - Yes/No
   - Low example: "The text covers professional soccer leagues" (mostly confined to Sports).
   - High example: "The text communicates a direct confrontation between opposing sides" (appears in Sports rivalries, World politics, Business competition).

q3. Does knowing the answer to this predicate (True/False) provide additional, relevant information within at least one (or more) of these categories that the category label alone does not capture?
   - Yes/No
   - Low example: "The text mentions a sports team name" (redundant within Sports; no new insight).
   - High example: "The text focuses on factual reporting rather than personal opinion" (distinguishes neutral vs. editorialized articles within Business or World).

q4. Does the predicate denote a human intention underlying the text, rather than a linguistic characteristic like the presence of a specific word, phrase, entity, or pattern?
   - Yes/No
   - Low example: "The text contains the word 'stock'" (surface linguistic feature).
   - High example: "The text communicates a direct confrontation between opposing sides" (captures argumentative intent).

q5. Is the predicate sufficiently precise and unambiguous that a diverse set of human readers would reliably agree (>80% agreement expected) on True/False labels for unseen documents without needing extra definitions or context?
   - Yes/No
   - Low-score examples: "The text is interesting", "The text denotes a shift in focus" (vague boundary; different readers interpret 'shift' differently).
   - High-score examples: "The text focuses on factual reporting rather than personal opinion", "The text communicates a direct confrontation between opposing sides" (clear conceptual duality; high expected agreement).

q6. Does this predicate have clear potential for real-world applications in information retrieval, content recommendation, or data organization (e.g., could it meaningfully filter/group news articles for users)?
   - Yes/No
   - Low example: "The text contains more than 500 words" (trivial length filter; limited insight).
   - High example: "The text focuses on factual reporting rather than personal opinion" (useful for recommending neutral news sources or filtering opinion pieces).

q7. Does this predicate denote a meaningful subset of exactly one of the topics (i.e., it is not redundant with the topic label, but it only applies within a single topic)?
    - Yes/No
    - Low example: "The text covers professional soccer leagues" (only relevant within Sports).
    - High example: "The text focuses on factual reporting rather than personal opinion" (applies across multiple topics).
    
    """

    agent = Agent(
        model=model,
        output_type=Evaluation,
        instructions=prompt,
        model_settings={
            "temperature": 0.2,
        },
    )
    result = agent.run_sync("")
    print("Evaluation used tokens:", result.usage().total_tokens)
    print("Input tokens:", result.usage().input_tokens)
    print("Output tokens:", result.usage().output_tokens)
    return result.output


def make_many_evals(
    role: str,
    question: str,
    labels: list[str],
    n_runs: int = 1,
) -> AverageEvaluation:
    q1_scores = []
    q2_scores = []
    q3_scores = []
    q4_scores = []
    q5_scores = []
    q6_scores = []
    q7_scores = []
    for _ in range(n_runs):
        eval_result = evaluate_question(
            role=role,
            question=question,
            labels=labels,
        )
        q1_scores.append(int(eval_result.q1))
        q2_scores.append(int(eval_result.q2))
        q3_scores.append(int(eval_result.q3))
        q4_scores.append(int(eval_result.q4))
        q5_scores.append(int(eval_result.q5))
        q6_scores.append(int(eval_result.q6))
        q7_scores.append(int(eval_result.q7))
        
    avg_eval = AverageEvaluation(
        q1=sum(q1_scores) / len(q1_scores),
        q2=sum(q2_scores) / len(q2_scores),
        q3=sum(q3_scores) / len(q3_scores),
        q4=sum(q4_scores) / len(q4_scores),
        q5=sum(q5_scores) / len(q5_scores),
        q6=sum(q6_scores) / len(q6_scores),
        q7=sum(q7_scores) / len(q7_scores),
    )
    
    return avg_eval


def get_all_nodes(node: TreeNode, max_level: int=4, level: int=1) -> list[TreeNode]:
    """Recursively get all nodes in the tree."""
    nodes = [node]
    if level < max_level:
        if node.left is not None:
            nodes.extend(get_all_nodes(node.left, max_level, level+1))
        if node.right is not None:
            nodes.extend(get_all_nodes(node.right, max_level, level+1))
    return nodes


from joblib import Parallel, delayed


def process_node(node, eval_questions, labels=["World", "Sports", "Business", "Sci/Tech"]) -> AverageEvaluation | None:
    if node.is_leaf():
        return None  # skip leaves

    print("Question:", node.question)
    print("Documents:", len(node.documents))
    print("-----")

    if eval_questions:
        eval_result = make_many_evals(
            role="senior news text analyst and topic modeling expert",
            question=node.question,
            labels=labels,
            n_runs=5,
        )
        print("Evaluation:", eval_result)
        print("=====")
        return eval_result
    return None


def get_dataset_from_filename(filename: str) -> str:
    if "ag_news" in filename or "agnews" in filename:
        return "agnews"
    elif "wikipedia" in filename:
        return "wikipedia"
    elif "bills" in filename:
        return "bills"
    elif "20_newsgroups" in filename:
        return "newsgroups"
    else:
        raise ValueError(f"Unknown dataset in filename: {filename}")

labels = {
    'agnews': ["World", "Sports", "Business", "Sci/Tech"],
    'wikipedia': ['Sports and recreation', 'Video games', 'Philosophy and religion', 'Music', 'Language and literature', 'Art and architecture', 'Media and drama', 'Engineering and technology', 'Geography and places', 'Mathematics', 'Agriculture, food, and drink', 'Warfare', 'Natural sciences', 'History', 'Social sciences and society'],
    'bills': ['Transportation', 'Domestic Commerce', 'International Affairs', 'Law and Crime', 'Public Lands', 'Housing', 'Foreign Trade', 'Environment', 'Immigration', 'Macroeconomics', 'Education', 'Civil Rights', 'Technology', 'Agriculture', 'Government Operations', 'Health', 'Labor', 'Social Welfare', 'Culture', 'Defense', 'Energy'],
    'newsgroups': ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
}



def process_file(input_fn: str, eval_questions: bool = False):
    input_path = Path(input_fn)
    with open(input_path, 'r') as f:
        json_data = f.read()
    loaded_tree = TreeNode.model_validate_json(json_data)
    nodes = get_all_nodes(loaded_tree)

    # Run in parallel across all nodes
    eval_results_ = Parallel(n_jobs=1, prefer="threads")(
        delayed(process_node)(node, eval_questions, labels=labels[get_dataset_from_filename(input_fn)]) for node in nodes)
    
    
    eval_results = [res for res in eval_results_ if res is not None]
    
    if eval_questions:
        # Convert eval results to a DataFrame
        df = pd.DataFrame([eval_result.model_dump() for eval_result in eval_results])
        df.to_csv(input_path.parent / f"eval_questions_{input_path.stem}.csv", index=False)
        print(df.mean().transpose().round(3).to_latex())


def run():
    args = read_input_arguments()
    inputs = args.input
    eval = args.eval

    for input_fn in inputs:
        process_file(input_fn, eval)
        exit()


def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trees")

    # Define arguments
    parser.add_argument("--input",
                        type=str,
                        nargs="+",
                        required=True,
                        help="Input file path")

    parser.add_argument("--eval",
                        default=False,
                        action='store_true',
                        help="Eval questions using LLM")

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    run()
