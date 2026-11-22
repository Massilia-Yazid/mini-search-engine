from rank_bm25 import BM25Okapi
from src.preprocessing import preprocess_text

def build_bm25_index(documents):
    """
    build a BM25 index from all documents
    """

    #extract document contents
    contents = documents["content"].tolist()

    #preprocess each document
    tokenized_corpus = [
        preprocess_text(content)
        for content in contents
    ]

    #build the BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

def search_bm25(query, documents, bm25, top_k=5):
    """
    search documents using BM25
    """
    #preprocess the query
    tokenized_query = preprocess_text(query)

    #calculate BM25 scores for all documents
    scores = bm25.get_scores(tokenized_query)

    #sort documents by score (descending)
    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = scores[index]

        #ignore documents with no positive score
        if score > 0:
            results.append({
                 "id": documents.iloc[index]["id"],
                "category": documents.iloc[index]["category"],
                "content": documents.iloc[index]["content"],
                "score": score
            })
    return results

