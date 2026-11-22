
"""
Mini Search Engine - Experiments and Evaluation

This script:
1. Loads the document collection.
2. Tests preprocessing.
3. Builds TF-IDF and BM25 indexes.
4. Tests both search algorithms.
5. Evaluates their performance using Precision@5.
"""

# ==========================================
# 1. IMPORTS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from src.preprocessing import preprocess_text

from src.tfidf_search import (
    build_tfidf_index,
    search_tfidf
)

from src.bm25_search import (
    build_bm25_index,
    search_bm25
)

from src.evaluation import (
    precision_at_k,
    evaluate_search_engine
)


# ==========================================
# 2. LOAD DOCUMENTS
# ==========================================

documents = pd.read_csv("data/documents.csv")

print("=" * 70)
print("MINI SEARCH ENGINE - EXPERIMENTS")
print("=" * 70)

print("\nTotal documents:", len(documents))


# ==========================================
# 3. TEST PREPROCESSING
# ==========================================

print("\n" + "=" * 70)
print("PREPROCESSING TEST")
print("=" * 70)

sample_text = documents.iloc[0]["content"]

print("\nOriginal text:")
print(sample_text)

print("\nProcessed text:")
print(preprocess_text(sample_text))


# ==========================================
# 4. BUILD SEARCH INDEXES
# ==========================================

print("\n" + "=" * 70)
print("BUILDING SEARCH INDEXES")
print("=" * 70)

# Build TF-IDF index
vectorizer, tfidf_matrix = build_tfidf_index(documents)

print("\nTF-IDF index created successfully.")

# Build BM25 index
bm25 = build_bm25_index(documents)

print("BM25 index created successfully.")


# ==========================================
# 5. DEFINE GROUND TRUTH
# ==========================================

queries = {

    "programming language": {1, 2, 3, 4, 5, 6},

    "artificial intelligence": {7, 8, 9, 10, 11, 12},

    "computer networks": {13, 14, 15, 16, 17, 18},

    "database management": {19, 20, 21, 22, 23},

    "algorithm data structure": {24, 25, 26, 27, 28, 29, 30}

}


# ==========================================
# 6. EVALUATE TF-IDF
# ==========================================

print("\n" + "=" * 70)
print("TF-IDF EVALUATION")
print("=" * 70)

tfidf_arguments = {
    "documents": documents,
    "vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix
}

tfidf_results, tfidf_mean = evaluate_search_engine(
    search_tfidf,
    queries,
    tfidf_arguments,
    k=5
)

for query, precision in tfidf_results.items():

    print(f"\nQuery: {query}")
    print(f"Precision@5: {precision:.2f}")

print("\nTF-IDF Mean Precision@5:", round(tfidf_mean, 4))


# ==========================================
# 7. EVALUATE BM25
# ==========================================

print("\n" + "=" * 70)
print("BM25 EVALUATION")
print("=" * 70)

bm25_arguments = {
    "documents": documents,
    "bm25": bm25
}

bm25_results, bm25_mean = evaluate_search_engine(
    search_bm25,
    queries,
    bm25_arguments,
    k=5
)

for query, precision in bm25_results.items():

    print(f"\nQuery: {query}")
    print(f"Precision@5: {precision:.2f}")

print("\nBM25 Mean Precision@5:", round(bm25_mean, 4))


# ==========================================
# 8. FINAL COMPARISON
# ==========================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print("\n{:<35} {:<15} {:<15}".format(
    "Query",
    "TF-IDF",
    "BM25"
))

print("-" * 70)

for query in queries:

    print("{:<35} {:<15.2f} {:<15.2f}".format(
        query,
        tfidf_results[query],
        bm25_results[query]
    ))

print("-" * 70)

print("{:<35} {:<15.2f} {:<15.2f}".format(
    "MEAN PRECISION@5",
    tfidf_mean,
    bm25_mean
))

print("\nExperiments and evaluation completed successfully!")
print("=" * 70)


# ==========================================
# 9. DATA VISUALIZATION WITH MATPLOTLIB
# ==========================================

# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)

# Prepare data
query_names = list(queries.keys())

tfidf_scores = [
    tfidf_results[query]
    for query in query_names
]

bm25_scores = [
    bm25_results[query]
    for query in query_names
]

# ==========================================
# GRAPH 1: Precision@5 per query
# ==========================================

x = np.arange(len(query_names))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(x - width / 2, tfidf_scores, width, label="TF-IDF")
plt.bar(x + width / 2, bm25_scores, width, label="BM25")

plt.xlabel("Search Queries")
plt.ylabel("Precision@5")
plt.title("TF-IDF vs BM25 - Precision@5 per Query")

plt.xticks(x, query_names, rotation=20, ha="right")
plt.ylim(0, 1.1)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("results/precision_comparison.png", dpi=300)
plt.show()


# ==========================================
# GRAPH 2: Mean Precision@5 comparison
# ==========================================

algorithms = ["TF-IDF", "BM25"]
mean_scores = [tfidf_mean, bm25_mean]

plt.figure(figsize=(7, 5))

plt.bar(algorithms, mean_scores)

plt.xlabel("Search Algorithms")
plt.ylabel("Mean Precision@5")
plt.title("Mean Precision@5 Comparison")

plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("results/mean_precision_comparison.png", dpi=300)
plt.show()

print("\nGraphs saved successfully in the results/ directory.")
