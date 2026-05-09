# Swiggy Restaurant Recommendation System — Project Report

---

## 1. Project Overview

This project builds an end-to-end restaurant recommendation system using real Swiggy restaurant data. Users specify their preferences (city, cuisine, budget, rating) through an interactive Streamlit web app and receive personalised restaurant recommendations powered by Cosine Similarity and K-Means Clustering.

---

## 2. Dataset Description

**Source file:** `swiggy.csv`

| Column         | Raw Type | Description                                      |
|----------------|----------|--------------------------------------------------|
| `id`           | int      | Unique Swiggy restaurant ID                      |
| `name`         | string   | Restaurant name                                  |
| `city`         | string   | City where the restaurant is located             |
| `rating`       | string   | Customer rating — e.g. `"4.2"`, `"--"`          |
| `rating_count` | string   | Number of ratings — e.g. `"50+ ratings"`         |
| `cost`         | string   | Cost for two — e.g. `"₹ 200"`, `"₹ 1,500"`      |
| `cuisine`      | string   | Comma-separated cuisine types                    |
| `lic_no`       | string   | FSSAI license number                             |
| `link`         | string   | Direct Swiggy URL                                |
| `address`      | string   | Full restaurant address                          |
| `menu`         | string   | Path to menu JSON file                           |

---

## 3. Data Cleaning and Preprocessing

### 3.1 Duplicate Removal
Fully duplicate rows were dropped with `pd.DataFrame.drop_duplicates()`.

### 3.2 Column Parsing

**`rating`**
Values like `"--"` and `"Too Few Ratings"` were converted to `NaN`. Only values in the range 1.0–5.0 were kept as valid floats; everything else became `NaN`.

**`rating_count`**
Strings like `"50+ ratings"` and `"1,000+ ratings"` were parsed by extracting the leading digit sequence using regex and removing commas. `"Too Few Ratings"` and similar non-numeric values became `NaN`.

**`cost`**
The rupee symbol (₹), commas, and whitespace were stripped using regex (`re.sub(r"[₹,\s]", "", val)`), and the result was cast to `float`.

### 3.3 Missing Value Handling
- Rows missing any of `name`, `city`, `cuisine`, `rating`, or `cost` were dropped (critical fields).
- `rating_count` nulls remaining after parsing were imputed with the column median.

### 3.4 Index Reset
After all cleaning, the index was reset with `drop=True` to produce a clean `0, 1, 2, …` integer index. This is critical: `cleaned_data.csv` and `encoded_data.csv` must share the exact same index for result mapping to work.

### 3.5 Feature Engineering
- **`cuisine_primary`**: The first token of the comma-separated `cuisine` field (e.g. `"North Indian,Chinese"` → `"North Indian"`). This reduces cardinality for OHE.

### 3.6 One-Hot Encoding
`city` and `cuisine_primary` were encoded with `sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")`. The fitted encoder was saved as `encoder.pkl`. The `handle_unknown="ignore"` setting ensures unseen values at query time produce an all-zero vector rather than raising an error.

### 3.7 Min-Max Scaling
The three numerical features were scaled to [0, 1]:

```
scaled = (value − min) / (max − min)
```

Scaling bounds come from the cleaned dataset and are re-applied at inference time using the same `cleaned_df` statistics.

### 3.8 Output Files

| File               | Description                                              |
|--------------------|----------------------------------------------------------|
| `cleaned_data.csv` | Human-readable cleaned Swiggy data                       |
| `encoded_data.csv` | Fully numerical feature matrix for ML                    |
| `encoder.pkl`      | Fitted `OneHotEncoder` (city + cuisine_primary)          |
| `kmeans_model.pkl` | Fitted `KMeans` model (generated on first kmeans call)   |

---

## 4. Recommendation Methodology

### 4.1 Query Vector Construction
When a user submits preferences, a single-row numpy array is built:

1. `rating`, `rating_count`, `cost` are normalised with the same min-max bounds from `cleaned_df` (values are clipped to the known range).
2. The chosen `city` and `cuisine_primary` (first token of the selected cuisine) are passed through `encoder.pkl` → OHE vector.
3. The numerical and OHE vectors are concatenated into a single `(1, n_features)` vector, where `n_features` matches `encoded_data.csv`.

### 4.2 Cosine Similarity (Method: `"cosine"`)
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```
The query vector is compared against all rows of `encoded_data.csv`. The top-N indices by descending similarity are retrieved and mapped to `cleaned_data.csv` to produce human-readable results.

**Pros:** No training required, interpretable, works well on sparse OHE vectors.

### 4.3 K-Means Clustering (Method: `"kmeans"`)
A `KMeans` model (30 clusters, `random_state=42`, `n_init=10`) is trained on `encoded_data.csv` once and cached as `kmeans_model.pkl`. At query time:

1. The query vector is assigned to its nearest cluster centroid.
2. Cosine similarity is computed only among restaurants in that cluster.
3. Top-N results are returned.
4. If the cluster is empty, the method falls back to global cosine similarity.

**Pros:** Scales better for large datasets; reduces the search space to relevant neighbourhood.

---

## 5. Streamlit Application (`app.py`)

### 5.1 Auto-Preprocessing
On first launch, if `cleaned_data.csv`, `encoded_data.csv`, or `encoder.pkl` are missing, the app automatically calls `data_preprocessing.main()` and restarts.

### 5.2 Sidebar Inputs
| Input | Type | Description |
|-------|------|-------------|
| City | Selectbox | Dropdown of all unique cities |
| Cuisine | Selectbox | Dropdown of all unique primary cuisines |
| Minimum Rating | Slider | 1.0–5.0 |
| Max Cost for Two | Slider | ₹ min–max from dataset |
| Min Number of Ratings | Slider | 0–max from dataset |
| Method | Radio | Cosine Similarity / K-Means Clustering |
| Results to Show | Slider | 5–20 |

### 5.3 Output
- **Summary metrics:** city, average rating, average cost, average match score.
- **Restaurant cards:** name, city, cuisine, address, rating, cost, match %, Swiggy link button.
- **Full results table** (expandable).
- **CSV download** button.

### 5.4 Welcome Panel
When no search has been run, a dataset overview shows total restaurants, cities, cuisines, average rating, and a preview of 10 rows.

---

## 6. Key Insights

- A large portion of Swiggy entries have `"--"` ratings (`"Too Few Ratings"`), indicating newly listed restaurants. These were excluded to ensure recommendation quality.
- Cuisine diversity is high; using the primary cuisine token reduces OHE dimensionality while retaining the most meaningful signal.
- Cosine similarity is particularly well-suited here because the OHE vectors are sparse and high-dimensional — angular distance is more informative than Euclidean distance in such spaces.
- K-Means with 30 clusters produces a meaningful neighbourhood structure given the dataset size, making per-cluster similarity search noticeably faster.

---

## 7. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean stale files and run preprocessing (FIRST TIME ONLY)
python setup_and_run.py

# 3. Launch the Streamlit app
streamlit run app.py
```

Or, to run preprocessing manually:
```bash
python data_preprocessing.py
streamlit run app.py
```

---

## 8. Project Structure

```
AIML_Swiggy_Restaurant_Recom_sys_MP4/
│
├── swiggy.csv                ← Raw Swiggy dataset (source)
│
├── data_preprocessing.py     ← Step 1: Clean, encode, save artefacts
├── recommendation.py         ← Step 2: Cosine & K-Means recommendation engine
├── app.py                    ← Step 3: Streamlit web application
├── setup_and_run.py          ← Cleanup + preprocessing launcher
├── requirements.txt          ← Python dependencies
├── report.md                 ← This report
│
├── cleaned_data.csv          ← [Generated] Cleaned restaurant data
├── encoded_data.csv          ← [Generated] Encoded feature matrix
├── encoder.pkl               ← [Generated] Fitted OneHotEncoder
└── kmeans_model.pkl          ← [Generated] Fitted KMeans model
```

---

*AIML Capstone Project MP4 — Swiggy Restaurant Recommendation System*
