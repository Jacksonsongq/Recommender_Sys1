import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import re

def compute_tfidf_matrix(restaurants):
    restaurants["Brief Description"] = restaurants["Brief Description"].fillna("")
    restaurants["Cuisine"] = restaurants["Cuisine"].fillna("")
    restaurants["Augmented Description"] = (
        restaurants["Brief Description"].str.strip().str.lower() + ". " +
        restaurants["Cuisine"].str.strip().str.lower()
    )
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 1)
    )
    X_full = vectorizer.fit_transform(restaurants["Augmented Description"])
    vocab = np.array(vectorizer.get_feature_names_out())
    return X_full, vocab, vectorizer

def top_term_restaurants(X_full, vocab, rest_names, terms):
    for term in terms:
        if term in vocab:
            col = X_full[:, vocab == term].toarray().ravel()
            idx = col.argmax()
            print(f"Highest TF-IDF for '{term}': {rest_names[idx]}  (score={col[idx]:.3f})")
        else:
            print(f"'{term}' not found in vocabulary")

def compute_distance_top100(X_full, vocab, rest_names):
    tfidf_sums = X_full.sum(axis=0).A1
    top100_idx = tfidf_sums.argsort()[-100:]
    top100_terms = vocab[top100_idx]
    X_100 = X_full[:, top100_idx].toarray()
    D = 1 - cosine_similarity(X_100)
    return X_100, top100_terms, D

def normalize_name(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKD", s).lower().strip().replace("’", "'"))

def print_pairwise_distance(D, rest_names, pairs):
    name2idx = {normalize_name(n): i for i, n in enumerate(rest_names)}
    print("\n=== TF-IDF cosine distances (top-100 terms) ===")
    for a, b in pairs:
        ia, ib = name2idx[normalize_name(a)], name2idx[normalize_name(b)]
        print(f"{a} ↔ {b} : {D[ia, ib]:.3f}")

def save_outputs(X_100, top100_terms, D, rest_names):
    sns.heatmap(D, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("TF-IDF cosine distance heat-map (63 restaurants)")
    plt.tight_layout()
    plt.savefig("tfidf_distance_heatmap.png", dpi=150)
    plt.close()

    pd.DataFrame(X_100, columns=top100_terms, index=rest_names).to_csv(
        "tfidf_top100_matrix.csv"
    )
    pd.DataFrame(D, index=rest_names, columns=rest_names).to_csv(
        "tfidf_distance_matrix.csv"
    )
    print("\nArtifacts saved: tfidf_top100_matrix.csv, tfidf_distance_matrix.csv, tfidf_distance_heatmap.png")
