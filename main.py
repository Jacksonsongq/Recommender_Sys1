import pandas as pd
from src import (
    compute_shrinkage,
    add_popularity_score,
    recommend,
    build_feature_matrix,
    compute_similarity_matrices,
    recommend_from_user,
    plot_top_k,
    jaccard,
    compute_tfidf_matrix,
    top_term_restaurants,
    compute_distance_top100,
    print_pairwise_distance,
    save_outputs
)

# === Load Data ===
file_path = "data/Evanston_Restaurant_Reviews.xlsx"
restaurants = pd.read_excel(file_path, sheet_name="Restaurants")
reviews = pd.read_excel(file_path, sheet_name="Reviews")

# === Shrinkage-Based Recommendation (Q4–Q5) ===
stats = compute_shrinkage(reviews)
stats = add_popularity_score(stats)

print("\n[Top Chinese Restaurants by Popularity Score]")
print(recommend(stats, restaurants, "Chinese"))

# === Content-Based Filtering (Q7–Q10) ===
cat_cols = ["Cuisine", "Average Cost", "Open After 8pm?"]
X = build_feature_matrix(restaurants, cat_cols)
D_euc, D_cos = compute_similarity_matrices(X)

user = "Willie Jacobsen"
fav, rec_euc = recommend_from_user(user, reviews, restaurants, D_euc)
_, rec_cos = recommend_from_user(user, reviews, restaurants, D_cos)

print(f"\n[Content-Based Filtering for {user} (Favorite: {fav})]")
print("\nTop-10 by Euclidean:\n", rec_euc)
print("\nTop-10 by Cosine:\n", rec_cos)

plot_top_k(rec_euc, user, "Euclidean", "figs/willie_euclidean.png")
plot_top_k(rec_cos, user, "Cosine", "figs/willie_cosine.png")

overlap = jaccard(rec_euc["Restaurant Name"], rec_cos["Restaurant Name"])
print(f"\nJaccard Overlap (Euclidean vs Cosine): {overlap:.2f}")

# === TF-IDF Similarity (Q6/Extension) ===
X_full, vocab, _ = compute_tfidf_matrix(restaurants)
rest_names = restaurants["Restaurant Name"].to_numpy()

top_term_restaurants(X_full, vocab, rest_names, ["cozy", "chinese"])

X_100, top100_terms, D = compute_distance_top100(X_full, vocab, rest_names)

pairs = [
    ("Burger King", "Edzo’s Burger Shop"),
    ("Burger King", "Oceanique"),
    ("Lao Sze Chuan", "Kabul House")
]
print_pairwise_distance(D, rest_names, pairs)
save_outputs(X_100, top100_terms, D, rest_names)
