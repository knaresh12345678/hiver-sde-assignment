import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/processed/amazon_english_pairs.csv"


print("Loading Amazon historical interactions...")

df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=["customer_text", "amazon_text"]
).reset_index(drop=True)

print(f"Historical interactions: {len(df):,}")


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

print("Building TF-IDF index...")

customer_vectors = vectorizer.fit_transform(
    df["customer_text"].astype(str)
)

print("Index ready.")


def retrieve_similar_cases(
    query,
    top_k=5,
    exclude_customer_text=None,
    exclude_customer_texts=None
):
    """
    Retrieve historically similar Amazon support cases.

    During evaluation, exact Golden Set customer messages can be
    excluded from the retrieval corpus to prevent data leakage.
    """

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        customer_vectors
    ).flatten()

    # ---------------------------------------------------------
    # Prevent evaluation leakage for one exact query
    # ---------------------------------------------------------

    if exclude_customer_text is not None:

        excluded = (
            df["customer_text"].astype(str)
            == str(exclude_customer_text)
        )

        scores[excluded.values] = -1

    # ---------------------------------------------------------
    # Prevent evaluation leakage for a set of queries
    # ---------------------------------------------------------

    if exclude_customer_texts is not None:

        excluded_texts = set(
            str(text)
            for text in exclude_customer_texts
        )

        excluded = (
            df["customer_text"].astype(str)
            .isin(excluded_texts)
        )

        scores[excluded.values] = -1

    # ---------------------------------------------------------
    # Select top results
    # ---------------------------------------------------------

    top_indices = scores.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    results["similarity"] = scores[top_indices]

    return results
