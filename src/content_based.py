import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def build_feature_matrix(restaurants, cat_cols):
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # for sklearn < 1.4
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe)
        ]), cat_cols)
    ])

    X = prep.fit_transform(restaurants[cat_cols])
    return X

def compute_similarity_matrices(X):
    D_euc = euclidean_distances(X)
    D_cos = 1 - cosine_similarity(X)
    return D_euc, D_cos

def recommend_from_user(user_name, reviews, restaurants, dist_matrix, k=10):
    rest_names = restaurants["Restaurant Name"].to_numpy()
    fav = (
        reviews.loc[reviews["Reviewer Name"] == user_name]
               .sort_values("Rating", ascending=False)
               .iloc[0]["Restaurant Name"]
    )
    fav_idx = np.where(rest_names == fav)[0][0]
    order = np.argsort(dist_matrix[fav_idx])[1 : k + 1]  # skip itself

    recs = (
        restaurants.iloc[order][["Restaurant Name", "Cuisine", "Average Cost"]]
        .assign(distance=dist_matrix[fav_idx, order])
        .reset_index(drop=True)
    )
    return fav, recs
