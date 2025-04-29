from .shrinkage import compute_shrinkage
from .recommender import add_popularity_score, recommend, recommend_by_cuisine
from .content_based import build_feature_matrix, compute_similarity_matrices, recommend_from_user
from .similarity_metrics import jaccard, plot_top_k
from .tf_idf_similarity import (
    compute_tfidf_matrix,
    top_term_restaurants,
    compute_distance_top100,
    print_pairwise_distance,
    save_outputs
)

