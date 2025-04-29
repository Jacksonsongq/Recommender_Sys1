import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_k(recs, user, dist_type, filename):
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=recs.iloc[::-1], 
        y="Restaurant Name", x="distance", palette="crest"
    )
    plt.title(f"{user}: top-10 nearest ({dist_type})")
    plt.xlabel("Distance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()

def jaccard(seq_a, seq_b):
    return len(set(seq_a) & set(seq_b)) / len(set(seq_a) | set(seq_b))
