import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def recommend_by_cuisine(stats, restaurants, cuisine, k=5):
    sub = (stats.merge(restaurants[["Restaurant Name", "Cuisine"]],
                       on="Restaurant Name")
                .query("Cuisine == @cuisine")
                .sort_values("shrunk", ascending=False)
                .head(k))
    return sub[["Restaurant Name", "shrunk", "num_reviews"]]


def add_popularity_score(stats):
    stats["pop_score"] = stats["shrunk"] + 0.1 * np.log1p(stats["num_reviews"])
    return stats


def recommend(stats, restaurants, cuisine, k=5):
    sub = (stats.merge(restaurants[["Restaurant Name", "Cuisine"]],
                       on="Restaurant Name")
                .query("Cuisine == @cuisine")
                .sort_values("pop_score", ascending=False)
                .head(k))
    return sub[["Restaurant Name", "avg_rating", "num_reviews", "pop_score"]]


def plot_shrinkage_effect(stats):
    top_pos = stats.nlargest(10, "delta")
    top_neg = stats.nsmallest(10, "delta")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_pos["Restaurant Name"], top_pos["delta"])
    ax.barh(top_neg["Restaurant Name"], top_neg["delta"], color="orange")
    ax.set_xlabel("Δ = Shrunk - Original")
    ax.set_title("Shrinkage Impact")
    plt.tight_layout()
    plt.show()
