import difflib
import re
import string
from typing import Any

import Levenshtein
import weave
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.translate import meteor
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from weave.scorers import Scorer

from scorers import (
    ResponseCorrectnessScorer,
    ResponseHelpfulnessScorer,
    ResponseRelevanceScorer,
)

wn.ensure_loaded()


def normalize_text(text: str) -> str:
    """
    Normalize the input text by lowercasing, removing punctuation, and extra whitespace.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    # Convert to lowercase
    if text is None:
        return "no output"
    text = text.lower()
    text = re.sub(r"[^\w\s\d]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@weave.op
def compute_diff(output: str, answer: str) -> float:
    """
    Compute the similarity ratio between the normalized model output and the expected answer.

    Args:
        output (str): The output generated by the model.
        answer (str): The expected answer.

    Returns:
        float: The similarity ratio between the normalized model output and the expected answer.
    """
    norm_output = normalize_text(output)
    norm_answer = normalize_text(answer)
    return difflib.SequenceMatcher(None, norm_output, norm_answer).ratio()


@weave.op
def compute_levenshtein(output: str, answer: str) -> float:
    """
    Compute the Levenshtein ratio between the normalized model output and the answer.

    Args:
        output (str): The output generated by the model.
        answer (str): The expected answer.

    Returns:
        float: The Levenshtein ratio between the normalized model output and the answer.
    """
    norm_output = normalize_text(output)
    norm_answer = normalize_text(answer)
    return Levenshtein.ratio(norm_output, norm_answer)


@weave.op
def compute_rouge(output: str, answer: str) -> float:
    """
    Compute the ROUGE-L F1 score between the normalized model output and the reference answer.

    Args:
        output (str): The model's generated output.
        answer (str): The reference answer.

    Returns:
        float: The ROUGE-L F1 score.
    """
    norm_output = normalize_text(output)
    norm_answer = normalize_text(answer)
    rouge = Rouge(metrics=["rouge-l"], stats="f")
    scores = rouge.get_scores(norm_output, norm_answer)
    return scores[0]["rouge-l"]["f"]


@weave.op
def compute_bleu(output: str, answer: str) -> float:
    """
    Compute the BLEU score between the normalized model output and the reference answer.

    Args:
        output (str): The generated output from the model.
        answer (str): The reference answer.

    Returns:
        float: The BLEU score between the normalized model output and the reference answer.
    """
    chencherry = SmoothingFunction()
    smoothing_function = chencherry.method2

    norm_output = normalize_text(output)
    norm_answer = normalize_text(answer)
    reference = word_tokenize(norm_answer)
    candidate = word_tokenize(norm_output)
    score = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
    return score


@weave.op
def compute_meteor(output: str, answer: str) -> float:
    """
    Compute the METEOR score between the normalized model output and the reference answer.

    Args:
        output (str): The model's generated output.
        answer (str): The reference answer.

    Returns:
        float: The METEOR score rounded to 4 decimal places.
    """
    norm_output = normalize_text(output)
    norm_answer = normalize_text(answer)
    reference = word_tokenize(norm_answer)
    candidate = word_tokenize(norm_output)
    meteor_score = round(meteor([candidate], reference), 4)
    return meteor_score


correctness_scorer = ResponseCorrectnessScorer()
helpfulness_scorer = ResponseHelpfulnessScorer()
relevance_scorer = ResponseRelevanceScorer()


class ResponseScorer(Scorer):
    @weave.op
    def score(
        self, output: dict[str, Any], question: str, answer: str
    ) -> dict[str, Any]:

        return {
            "diff": compute_diff(output.get("answer", ""), answer),
            "levenshtein": compute_levenshtein(output.get("answer", ""), answer),
            "rouge": compute_rouge(output.get("answer", ""), answer),
            "bleu": compute_bleu(output.get("answer", ""), answer),
            "meteor": compute_meteor(output.get("answer", ""), answer),
            "correctness": correctness_scorer.score(
                input=question, output=output.get("answer", ""), context=answer
            ),
            "helpfulness": helpfulness_scorer.score(
                input=question, output=output.get("answer", ""), context=answer
            ),
            "relevance": relevance_scorer.score(
                input=question, output=output.get("answer", ""), context=answer
            ),
        }
