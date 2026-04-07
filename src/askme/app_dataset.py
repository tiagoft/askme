import os
from pydantic import BaseModel

from .rtp.nli import NLIWithChunkingAndPooling
from .preprocess_pdf import get_document
import re
from tqdm import tqdm


def get_all_pdf_files_in_directory(directory: str) -> list[str]:
    """
    Recursively searches for all PDF files in the specified directory and its subdirectories.
    
    Args:
        directory (str): The path to the directory to search.
    
    Returns:
        list[str]: A list of file paths to all found PDF files.
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


class Configuration(BaseModel):
    """
    A Pydantic model representing the configuration for the application.
    """
    keywords: list[str] | None = None
    regex: list[str] | None = None
    nli_hypotheses: list[str] | None = None
    llm_hypotheses: list[str] | None = None


class Answer(BaseModel):
    """
    A Pydantic model representing an answer extracted from a document.
    """
    input: str
    output: float | int | bool | str | list | None
    method: str


class AllAnswers(BaseModel):
    """
    A Pydantic model representing all answers extracted from a document.
    """
    filename: str
    keywords: list[Answer]
    regex: list[Answer]
    nli_hypotheses: list[Answer]
    llm_hypotheses: list[Answer]


def app_search_on_directory(
    directory: str,
    config_file: str,
    n_max: int | None = None,
):
    """
    Runs the search for keywords and regex patterns on all PDF files in the specified directory based on the provided configuration file.
    
    Args:
        directory (str): The path to the directory containing PDF files.
        config_file (str): The path to the configuration file in TOML format.
    
    Returns:
        list[AllAnswers]: A list of AllAnswers objects containing the results for each processed file.
    """
    file_paths = get_all_pdf_files_in_directory(directory)
    config = load_configuration_file(config_file)
    if n_max is not None:
        file_paths = file_paths[:n_max]
    results = run_seach_on_files(file_paths, config)
    return results

def app_search_on_table(
    file_path: str,
    column_name: str,
    config_file: str,
    n_max: int | None = None,
):
    """
    Runs the search for keywords and regex patterns on a specified column of a table based on the provided configuration file.
    
    Args:
        file_path (str): The path to the file containing the table (e.g., CSV, Excel).
        column_name (str): The name of the column to search within.
        config_file (str): The path to the configuration file in TOML format.
    
    Returns:
        list[AllAnswers]: A list of AllAnswers objects containing the results for each processed entry in the column.
    """
    config = load_configuration_file(config_file)
    results = run_seach_on_table(file_path, column_name, config)
    return results


def load_configuration_file(config_path: str) -> dict:
    """
    Loads a configuration file in TOML format.
    
    Args:
        config_path (str): The path to the configuration file.
    
    Returns:
        dict: The loaded configuration as a dictionary.
    """
    import toml
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config


def find_keywords_in_document(document: str,
                              keywords: list[str]) -> list[Answer]:
    """
    Finds the presence of specified keywords in a document.
    
    Args:
        document (str): The text of the document to search.
        keywords (list[str]): A list of keywords to find in the document.
    
    Returns:
        list[Answer]: A list of Answer objects indicating the presence of each keyword.
    """
    answers = []
    for keyword in keywords:
        found = keyword in document
        answers.append(Answer(input=keyword, output=found, method="keyword"))
    return answers


def find_regex_patterns_in_document(document: str,
                                    regex_patterns: list[str]) -> list[Answer]:
    """
    Finds the presence of specified regex patterns in a document.
    
    Args:
        document (str): The text of the document to search.
        regex_patterns (list[str]): A list of regex patterns to find in the document.
    
    Returns:
        list[Answer]: A list of Answer objects indicating the presence of each regex pattern.
    """
    answers = []
    for pattern in regex_patterns:
        match = re.search(pattern, document)
        found_str = match.group(0) if match else False
        answers.append(Answer(input=pattern, output=found_str, method="regex"))
    return answers


def run_seach_on_files(
    file_paths: list[str],
    config: Configuration,
) -> list[AllAnswers]:
    """
    Runs the search for keywords and regex patterns on a list of files based on the provided configuration.
    
    Args:
        file_paths (list[str]): A list of file paths to process.
        config (Configuration): The configuration containing keywords and regex patterns.
    
    Returns:
        list[AllAnswers]: A list of AllAnswers objects containing the results for each file.
    """
    if isinstance(config, dict):
        config = Configuration(**config)

    if not isinstance(config, Configuration):
        raise ValueError(
            "Config must be a Configuration object or a dict that can be parsed into one."
        )

    if config.nli_hypotheses is not None:
        nli_model = NLIWithChunkingAndPooling(disable_tqdm=True)
    else:
        nli_model = None

    all_answers = []
    for path in tqdm(file_paths):
        try:
            if path.endswith('.pdf'):
                document, _ = get_document(path)
            else:
                with open(path, 'r') as f:
                    document = f.read()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            continue

        this_answer = _run_recipe_on_doc(
            document,
            config,
            nli_model,
            path,
        )
        all_answers.append(this_answer)

    return all_answers


def run_seach_on_table(
    file_path: str,
    column_name: str,
    config: Configuration,
) -> list[AllAnswers]:
    """
    Runs the search for keywords and regex patterns on a list of files based on the provided configuration.
    
    Args:
        file_path (str): The path to the file to process.
        column_name (str): The name of the column to search.
        config (Configuration): The configuration containing keywords and regex patterns.
    
    Returns:
        list[AllAnswers]: A list of AllAnswers objects containing the results for each file.
    """
    import pandas as pd
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

    data = df[column_name].astype(str).tolist()

    if isinstance(config, dict):
        config = Configuration(**config)

    if not isinstance(config, Configuration):
        raise ValueError(
            "Config must be a Configuration object or a dict that can be parsed into one."
        )

    if config.nli_hypotheses is not None:
        nli_model = NLIWithChunkingAndPooling(disable_tqdm=True)
    else:
        nli_model = None

    all_answers = []

    for document in tqdm(data):
        this_answer = _run_recipe_on_doc(
            document,
            config,
            nli_model,
            file_path,
        )
        all_answers.append(this_answer)

    return all_answers


def _run_recipe_on_doc(
    document: str,
    config: Configuration,
    nli_model: NLIWithChunkingAndPooling | None = None,
    file_path: str = "",
):
    if config.keywords is not None:
        keywords_answers = find_keywords_in_document(document, config.keywords)
    else:
        keywords_answers = []

    if config.regex is not None:
        regex_answers = find_regex_patterns_in_document(document, config.regex)
    else:
        regex_answers = []

    nli_answers = []
    if nli_model is not None:
        for hypothesis in config.nli_hypotheses:
            nli_result = nli_model([document], hypothesis)[0]
            nli_answers.append(
                Answer(input=hypothesis, output=nli_result, method="nli"))

    return AllAnswers(
        filename=file_path,
        keywords=keywords_answers,
        regex=regex_answers,
        nli_hypotheses=nli_answers,
        llm_hypotheses=[],
    )
