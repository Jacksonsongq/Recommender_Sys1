import numpy as np
import pandas as pd

def compute_shrinkage(df):
    stats = (df.groupby("Restaurant Name")["Rating"]
               .agg(avg_rating="mean", num_reviews="count")
               .reset_index())

    mu_s = stats["avg_rating"].mean()
    N_mu = stats["num_reviews"].mean()

    stats["alpha"] = (stats["num_reviews"] / N_mu).clip(upper=1)
    stats["shrunk"] = (1 - stats["alpha"]) * mu_s + stats["alpha"] * stats["avg_rating"]
    stats["delta"] = stats["shrunk"] - stats["avg_rating"]

    return stats
