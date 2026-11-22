"""
Evaluation module for the Mini Search Engine

This module evaluates search algorithms using Precision@k

"""


def precision_at_k(results, relevant_docs, k=5):
    """
    Calculate precision@k

    Parameters:
       results: List of retrieved documents, already ranked
       relevant_docs: Set of relevant document IDs
       k: Number of top results to evaluate

    Returns:
       Precision score between 0 and 1
    """
    if k <= 0:
        raise ValueError("k must be grater than zero.")
    # keep only the first K results
    top_k_results = results[:k]

    # count relevant documents among the retrieved results
    relevant_retrieved = sum(
        1
        for result in top_k_results
        if int(result["id"]) in relevant_docs
    )
    # calculate precision@k
    precision = relevant_retrieved / k

    return precision


def evaluate_search_engine(search_function, queries, search_arguments, k=5):
    """
    Evaluate a search engine on multiple queries

    Returns:
       A dictionary containing Precision@k for each query
       and the average Precision@k
    """
    results_by_query = {}

    for query, relevant_docs in queries.items():

        # execute the search
        results = search_function(
            query,
            **search_arguments,
            top_k=k
        )

        # calculate Precision@k

        precision = precision_at_k(
            results,
            relevant_docs,
            k
        )

        results_by_query[query] = precision

    # calculate mean Precision@k
    if results_by_query:
        mean_precision = (
            sum(results_by_query.values()) / len(results_by_query)
        )
    else:
        mean_precision = 0.0

    return results_by_query, mean_precision
