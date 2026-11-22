from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text

def build_tfidf_index(documents):
    """
    convert all documents into TF-IDF vectors
    """
    #extract document contents
    contents = documents["content"].tolist()
    #create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        token_pattern=None,
        lowercase=False
    )
    #Build the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(contents)

    return vectorizer, tfidf_matrix

def search_tfidf(query, documents, vectorizer, tfidf_matrix,top_k=5):
    """
    search documents using TF-IDF and cosine similarity
    """
    #convert the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    #calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

    #sort documents by similarity score (descending)
    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = similarities[index]
        #ignore documents with no similarity
        if score > 0:
            results.append({
                "id": documents.iloc[index]["id"],
                "category": documents.iloc[index]["category"],
                "content": documents.iloc[index]["content"],
                "score": score
            }
            )
    return results
