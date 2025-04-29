# 🍽️ Restaurant Recommender System – STAT 415 (Northwestern)

This project explores four fundamental approaches to building a restaurant recommendation engine using real-world survey data from Evanston, IL.

**Course**: STAT 415 – Machine Learning  
**Deliverable**: PDF report + executable Python package (this repo)  
**Language**: Python 3.10+  
**Data**: `Evanston_Restaurant_Reviews.xlsx`

---

## 📌 Approaches Implemented

1. **Exploratory Data Analysis (EDA)**  
   Understand user demographics, restaurant metadata, and rating distributions.  
   → See: `src/eda.py`

2. **Popularity-Based Recommendation**  
   Includes average rating and empirical Bayes shrinkage (weighted by review count).  
   → See: `src/shrinkage.py`, `src/recommender.py`

3. **Content-Based Filtering**  
   Recommend restaurants based on numeric & categorical embeddings (e.g., cuisine, cost, open hours).  
   → See: `src/content_based.py`

4. **Natural-Language TF-IDF Analysis**  
   Compare restaurant descriptions using TF-IDF and cosine distance.  
   → See: `src/tf_idf_similarity.py`

---

## 📂 Project Structure
## 📁 Project Structure

Recommender_Sys1/
├── data/                      # Raw data files
│   └── Evanston_Restaurant_Reviews.xlsx
├── figs/                      # Output visualizations
├── src/                       # Core Python modules
│   ├── __init__.py
│   ├── eda.py
│   ├── shrinkage.py
│   ├── recommender.py
│   ├── content_based.py
│   ├── tf_idf_similarity.py
│   └── similarity_metrics.py
├── main.py        # Run full pipeline (for demo)
├── README.md                  # Project documentation



