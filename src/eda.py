import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

FILE    = "Evanston Restaurant Reviews.xlsx"     # change here if the name differs
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

restaurants = pd.read_excel(FILE, sheet_name="Restaurants")
reviews     = pd.read_excel(FILE, sheet_name="Reviews")

df = reviews.merge(
        restaurants[["Restaurant Name", "Cuisine"]],
        on="Restaurant Name", how="left"
     )
print(f"[INFO] merged DataFrame shape: {df.shape}")

# missing-value audit 
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

# distributions 
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

# clustering grid search 
FEATURES = [
    "Birth Year", "Weight (lb)", "Has Children?", "Marital Status",
    "Preferred Mode of Transport", "Average Amount Spent"
]

cat_cols = [c for c in FEATURES if df[c].dtype == "O"]
num_cols = list(set(FEATURES) - set(cat_cols))

# OneHotEncoder
try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:          
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

# K-Means & Agglomerative clustering
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

# DBSCAN parameter grid 
for eps in (4, 6, 8, 10, 12):
    for ms in (3, 5, 8):
        db = DBSCAN(eps=eps, min_samples=ms).fit(X)
        labels = db.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k <= 1:    
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
display(res)                         
res.to_csv("cluster_scan_results.csv", index=False)
print("Grid search finished – results saved to cluster_scan_results.csv")

# Profile Clusters for KMeans
def profile_kmeans(X, df_work, k_list=[2, 3]):
    for k in k_list:
        print(f"\n=== KMeans clustering: k = {k} ===")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        
        df_work["cluster"] = labels
        
        # Only use columns that exist
        profile_cols = ["Rating", "Birth Year", "Weight (lb)"]
        available_cols = [col for col in profile_cols if col in df_work.columns]
        
        profile = (df_work
                   .groupby("cluster")[available_cols]
                   .mean()
                   .round(2))
        
        display(profile)
        profile.to_csv(f"profile_kmeans_k{k}.csv")

X_for_kmeans = prep.transform(work[FEATURES])
profile_kmeans(X_for_kmeans, work.copy())


