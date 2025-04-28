# ===== Evanston Restaurant Reviews – Complete EDA (Q1–Q3) =====
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------- basic settings ---------
FILE    = "Evanston Restaurant Reviews.xlsx"     # change here if the name differs
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# --------- 1. load & merge ---------
restaurants = pd.read_excel(FILE, sheet_name="Restaurants")
reviews     = pd.read_excel(FILE, sheet_name="Reviews")

df = reviews.merge(
        restaurants[["Restaurant Name", "Cuisine"]],
        on="Restaurant Name", how="left"
     )
print(f"[INFO] merged DataFrame shape: {df.shape}")

# --------- Q1. missing-value audit ---------
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if missing.empty:
    print("No missing values detected")
else:
    pct = (missing / len(df)).round(3)
    print("=== Top 15 columns by missing count ===")
    display(pd.concat([missing, pct.rename("ratio")], axis=1).head(15))
    
    # bar-plot of missing ratio
    (pct * 100).plot(kind="barh",
                     figsize=(6, max(3, 0.3 * len(missing))))
    plt.xlabel("% missing")
    plt.title("Missing-value ratio")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/missing_values.png", dpi=120)
    plt.close()

# --------- Q2. distributions (histograms / bar charts) ---------
HIST_VARS = {
    "Cuisine": "cat",
    "Average Amount Spent": "cat",
    "Vegetarian?": "cat",
    "Has Children?": "cat",
    "Weight (lb)": "num",
    "Preferred Mode of Transport": "cat",
}
for col, typ in HIST_VARS.items():
    if col not in df.columns:
        print(f"[WARN] column not found: {col}")
        continue
    plt.figure(figsize=(5, 3))
    if typ == "num":
        sns.histplot(df[col].dropna(), bins=30)
    else:
        sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()
    fname = col.replace(" ", "_").replace("?", "")
    plt.savefig(f"{FIG_DIR}/{fname}.png", dpi=120)
    plt.close()

print(f"[INFO] all distribution plots saved → {FIG_DIR}/")

# --------- Q3. clustering grid search (KMeans / Agglo / DBSCAN) ---------
FEATURES = [
    "Birth Year", "Weight (lb)", "Has Children?", "Marital Status",
    "Preferred Mode of Transport", "Average Amount Spent"
]

cat_cols = [c for c in FEATURES if df[c].dtype == "O"]
num_cols = list(set(FEATURES) - set(cat_cols))

# OneHotEncoder – new & old scikit-learn compatibility
try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:          # sklearn < 1.4
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ]), cat_cols)
])

work = df[FEATURES + ["Rating"]].copy()
X = prep.fit_transform(work[FEATURES])

records = []

# --- K-Means & Agglomerative clustering ---
for algo, ctor in [("kmeans", KMeans), ("agglo", AgglomerativeClustering)]:
    for k in range(2, 7):
        if algo == "kmeans":
            labels = ctor(n_clusters=k, random_state=42,
                           n_init="auto").fit_predict(X)
        else:
            labels = ctor(n_clusters=k, linkage="ward").fit_predict(X)
            
        sil = silhouette_score(X, labels)
        work["tmp"] = labels
        avg = (work.groupby("tmp")["Rating"]
                   .mean().round(2).tolist())
        records.append(dict(
            algo=algo, param=k, clusters=k,
            silhouette=round(sil, 3), avg_ratings=avg
        ))

# --- DBSCAN parameter grid ---
for eps in (8, 10, 12):
    for ms in (5, 10):
        db = DBSCAN(eps=eps, min_samples=ms).fit(X)
        labels = db.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k <= 1:    # ignore trivial clustering
            continue
        sil = silhouette_score(X, labels)
        work["tmp"] = labels
        avg = (work[work["tmp"] != -1]
                   .groupby("tmp")["Rating"]
                   .mean().round(2).tolist())
        records.append(dict(
            algo="dbscan",
            param=f"eps={eps},min={ms}",
            clusters=k, silhouette=round(sil, 3),
            avg_ratings=avg
        ))

res = (pd.DataFrame(records)
         .sort_values(["algo", "silhouette"], ascending=[True, False]))
display(res)                         # quick inspection
res.to_csv("cluster_scan_results.csv", index=False)
print("Grid search finished – results saved to cluster_scan_results.csv")
