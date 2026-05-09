"""
=============================================================
  Swiggy Restaurant Recommendation System
  Script : recommendation.py

  Public API:
    get_recommendations(city, cuisine, rating, rating_count,
                        cost, method="cosine", top_n=10)
      → pd.DataFrame  (rows from cleaned_data.csv + similarity_score)

  Methods:
    "cosine"  – Cosine Similarity (global, no training needed)
    "kmeans"  – K-Means cluster → Cosine Similarity within cluster
=============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV  = os.path.join(BASE_DIR, "cleaned_data.csv")
ENC_CSV    = os.path.join(BASE_DIR, "encoded_data.csv")
ENC_PKL    = os.path.join(BASE_DIR, "encoder.pkl")
KMEANS_PKL = os.path.join(BASE_DIR, "kmeans_model.pkl")


# ── Load artefacts ────────────────────────────────────────────────────────────
def load_artifacts():
    """Load and return (cleaned_df, encoded_df, encoder)."""
    cleaned_df = pd.read_csv(CLEAN_CSV, index_col=0)
    encoded_df = pd.read_csv(ENC_CSV,   index_col=0)
    with open(ENC_PKL, "rb") as f:
        encoder = pickle.load(f)
    return cleaned_df, encoded_df, encoder


# ── Build query vector ────────────────────────────────────────────────────────
def build_query_vector(encoder, encoded_df, cleaned_df,
                       city, cuisine, rating, rating_count, cost):
    """
    Produces a (1, n_features) numpy array that matches encoded_data.csv layout:
        [rating_norm, rating_count_norm, cost_norm,  <OHE city cols>,  <OHE cuisine cols>]
    """

    # ── Numerical: same min-max as preprocessing ──────────────────────────────
    def norm(value, col):
        cmin = cleaned_df[col].min()
        cmax = cleaned_df[col].max()
        if cmax > cmin:
            val = np.clip(float(value), cmin, cmax)
            return (val - cmin) / (cmax - cmin)
        return 0.0

    num_vec = np.array([[
        norm(rating,       "rating"),
        norm(rating_count, "rating_count"),
        norm(cost,         "cost"),
    ]])                                          # shape (1, 3)

    # ── Categorical: OHE ─────────────────────────────────────────────────────
    cuisine_primary = str(cuisine).split(",")[0].strip()
    input_df = pd.DataFrame(
        [[city, cuisine_primary]],
        columns=["city", "cuisine_primary"]
    )
    ohe_vec = encoder.transform(input_df)        # shape (1, n_ohe)

    # ── Concatenate ───────────────────────────────────────────────────────────
    query = np.concatenate([num_vec, ohe_vec], axis=1)  # shape (1, n_total)

    # Width safety: align to encoded_df column count
    expected = encoded_df.shape[1]
    if query.shape[1] < expected:
        pad   = np.zeros((1, expected - query.shape[1]))
        query = np.concatenate([query, pad], axis=1)
    elif query.shape[1] > expected:
        query = query[:, :expected]

    return query


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def recommend_cosine(query_vec, cleaned_df, encoded_df, top_n=10):
    sims    = cosine_similarity(query_vec, encoded_df.values).flatten()
    top_idx = np.argsort(sims)[::-1][:top_n]
    result  = cleaned_df.iloc[top_idx].copy()
    result["similarity_score"] = np.round(sims[top_idx], 4)
    return result.reset_index(drop=True)


# ── K-Means ───────────────────────────────────────────────────────────────────
def _get_kmeans(encoded_df, n_clusters=30):
    """Load cached KMeans or train, save, and return a new one."""
    if os.path.exists(KMEANS_PKL):
        with open(KMEANS_PKL, "rb") as f:
            km = pickle.load(f)
        # Validate: labels must cover the current encoded_df
        if hasattr(km, "labels_") and len(km.labels_) == len(encoded_df):
            return km
        print("[KMEANS] Cached model is stale — retraining…")

    print(f"[KMEANS] Training KMeans with {n_clusters} clusters…")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(encoded_df.values)
    with open(KMEANS_PKL, "wb") as f:
        pickle.dump(km, f)
    print("[KMEANS] Model trained and saved.")
    return km


def recommend_kmeans(query_vec, cleaned_df, encoded_df, top_n=10):
    km      = _get_kmeans(encoded_df)
    cluster = int(km.predict(query_vec)[0])
    indices = np.where(km.labels_ == cluster)[0]

    if len(indices) == 0:
        # Fallback: global cosine
        return recommend_cosine(query_vec, cleaned_df, encoded_df, top_n)

    cluster_enc = encoded_df.iloc[indices]
    sims        = cosine_similarity(query_vec, cluster_enc.values).flatten()
    top_local   = np.argsort(sims)[::-1][:top_n]
    top_global  = indices[top_local]

    result = cleaned_df.iloc[top_global].copy()
    result["similarity_score"] = np.round(sims[top_local], 4)
    return result.reset_index(drop=True)


# ── Public API ────────────────────────────────────────────────────────────────
def get_recommendations(
    city,
    cuisine,
    rating       = 4.0,
    rating_count = 50.0,
    cost         = 300.0,
    method       = "cosine",
    top_n        = 10
):
    """
    Parameters
    ----------
    city         : str   – city name (must exist in cleaned_data.csv)
    cuisine      : str   – cuisine preference (primary token used)
    rating       : float – desired minimum rating (1.0–5.0)
    rating_count : float – approximate number of ratings
    cost         : float – approximate cost for two (₹)
    method       : str   – "cosine" or "kmeans"
    top_n        : int   – number of results to return

    Returns
    -------
    pd.DataFrame with top_n rows from cleaned_data.csv + similarity_score
    """
    cleaned_df, encoded_df, encoder = load_artifacts()

    query_vec = build_query_vector(
        encoder, encoded_df, cleaned_df,
        city, cuisine, rating, rating_count, cost
    )

    if method == "kmeans":
        return recommend_kmeans(query_vec, cleaned_df, encoded_df, top_n)
    else:
        return recommend_cosine(query_vec, cleaned_df, encoded_df, top_n)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Recommendation Engine — Self Test")
    print("=" * 60)

    cleaned_df, _, _ = load_artifacts()
    sample = cleaned_df.iloc[0]
    print(f"Query sample row:\n{sample[['name','city','cuisine','rating','cost']]}\n")

    for m in ["cosine", "kmeans"]:
        print(f"\n── Method: {m} ──")
        recs = get_recommendations(
            city         = sample["city"],
            cuisine      = sample["cuisine"],
            rating       = float(sample["rating"]),
            rating_count = float(sample["rating_count"]),
            cost         = float(sample["cost"]),
            method       = m,
            top_n        = 5
        )
        print(recs[["name", "city", "cuisine", "rating", "cost", "similarity_score"]])
