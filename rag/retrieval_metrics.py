from typing import Any

import numpy as np
import weave
from weave.flow.scorer import Scorer

from scorers import DocumentRelevanceScorer


@weave.op
def compute_hit_rate(
    output: list[dict[str, Any]], contexts: list[dict[str, Any]]
) -> float:
    """
    Calculate the hit rate (precision) for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The hit rate (precision).

    The hit rate (precision) measures the proportion of retrieved documents that are relevant.

    This metric is useful for assessing the accuracy of the retrieval system by determining the relevance of the
    retrieved documents.
    """
    search_results = [doc["source"] for doc in output]
    relevant_sources = [
        context["source"] for context in contexts if context["relevance"] != 0
    ]

    # Calculate the number of relevant documents retrieved
    relevant_retrieved = sum(
        1 for source in search_results if source in relevant_sources
    )

    # Calculate the hit rate (precision)
    hit_rate = relevant_retrieved / len(search_results) if search_results else 0.0

    return hit_rate


@weave.op
def compute_mrr(output: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The MRR score for the given query.

    MRR measures the rank of the first relevant document in the result list.

    If no relevant document is found, MRR is 0.

    This metric is useful for evaluating systems where there is typically one relevant document
    and the user is interested in finding that document quickly.
    """
    relevant_sources = [
        context["source"] for context in contexts if context["relevance"] != 0
    ]

    mrr_score = 0
    for rank, result in enumerate(output, 1):
        if result["source"] in relevant_sources:
            mrr_score += 1 / rank

    if mrr_score == 0:
        return 0.0
    else:
        return mrr_score / len(output)


@weave.op
def compute_ndcg(output: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.
                - 'relevance': The relevance score of the document (0, 1, or 2).

    Returns:
        float: The NDCG score for the given query.
    """
    relevance_map = {context["source"]: context["relevance"] for context in contexts}

    dcg = 0.0
    idcg = 0.0

    for i, result in enumerate(output):
        rel = relevance_map.get(result["source"], 0)
        dcg += (2**rel - 1) / np.log2(i + 2)

    sorted_relevances = sorted(
        [context["relevance"] for context in contexts], reverse=True
    )
    for i, rel in enumerate(sorted_relevances):
        idcg += (2**rel - 1) / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


@weave.op
def compute_map(output: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> float:
    """
    Calculate the Mean Average Precision (MAP) for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The MAP score for the given query.

    MAP provides a single-figure measure of quality across recall levels.
    For a single query, it's equivalent to the Average Precision (AP).


    Where:
    - n is the number of retrieved documents
    - P(k) is the precision at cut-off k in the list
    - rel(k) is an indicator function: 1 if the item at rank k is relevant, 0 otherwise
    MAP considers both precision and recall, as well as the ranking of relevant documents.

    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }

    num_relevant = 0
    sum_precision = 0.0

    for i, result in enumerate(output):
        if result["source"] in relevant_sources:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    average_precision = sum_precision / len(relevant_sources)
    return average_precision


@weave.op
def compute_precision(
    output: list[dict[str, Any]], contexts: list[dict[str, Any]]
) -> float:
    """
    Calculate the Precision for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The Precision score for the given query.

    Precision measures the proportion of retrieved documents that are relevant.
    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }
    retrieved_sources = {result["source"] for result in output}

    relevant_retrieved = relevant_sources & retrieved_sources

    precision = (
        len(relevant_retrieved) / len(retrieved_sources) if retrieved_sources else 0.0
    )
    return precision


# Recall
@weave.op
def compute_recall(
    output: list[dict[str, Any]], contexts: list[dict[str, Any]]
) -> float:
    """
    Calculate the Recall for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The Recall score for the given query.

    Recall measures the proportion of relevant documents that are retrieved.
    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }
    retrieved_sources = {result["source"] for result in output}

    relevant_retrieved = relevant_sources & retrieved_sources

    recall = (
        len(relevant_retrieved) / len(relevant_sources) if relevant_sources else 0.0
    )
    return recall


# F1 Score
@weave.op
def compute_f1_score(
    output: list[dict[str, Any]], contexts: list[dict[str, Any]]
) -> float:
    """
    Calculate the F1-Score for a single query.

    Args:
        output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The F1-Score for the given query.

    F1-Score is the harmonic mean of Precision and Recall.
    """
    precision = compute_precision(output, contexts)
    recall = compute_recall(output, contexts)

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


context_relevance_scorer = DocumentRelevanceScorer()


class RetrievalScorer(Scorer):
    @weave.op
    async def score(
        self,
        question: str,
        output: list[dict[str, Any]],
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:

        output_sources = [{"source": doc["uri"]} for doc in output]
        output_texts = [doc["text"] for doc in output]
        return {
            "hit_rate": compute_hit_rate(output_sources, contexts),
            "mrr": compute_mrr(output_sources, contexts),
            "ndcg": compute_ndcg(output_sources, contexts),
            "map": compute_map(output_sources, contexts),
            "precision": compute_precision(output_sources, contexts),
            "recall": compute_recall(output_sources, contexts),
            "f1_score": compute_f1_score(output_sources, contexts),
            "relevance": await context_relevance_scorer.score(
                input=question, output=output_texts
            ),
        }
