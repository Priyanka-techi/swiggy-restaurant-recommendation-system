"""
=============================================================
  Swiggy Restaurant Recommendation System
  Script : app.py  –  Streamlit Application

  Run:
      streamlit run app.py
=============================================================
"""

import os
import sys
import pandas as pd
import streamlit as st

# ── Resolve base directory ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CLEAN_CSV  = os.path.join(BASE_DIR, "cleaned_data.csv")
ENC_CSV    = os.path.join(BASE_DIR, "encoded_data.csv")
ENC_PKL    = os.path.join(BASE_DIR, "encoder.pkl")

# ── Auto-preprocess if artefacts are missing ──────────────────────────────────
_artefacts = [CLEAN_CSV, ENC_CSV, ENC_PKL]
if not all(os.path.exists(p) for p in _artefacts):
    st.info("⚙️ First run: generating preprocessed artefacts — please wait…")
    import data_preprocessing
    data_preprocessing.main()
    st.rerun()

from recommendation import get_recommendations   # noqa: E402 (import after preprocessing)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a cleaner look ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .restaurant-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load cleaned data (cached) ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_cleaned() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV, index_col=0)


df = load_cleaned()

# ── Build filter lists ────────────────────────────────────────────────────────
all_cities = sorted(df["city"].dropna().unique().tolist())

all_cuisines = sorted(set(
    token.strip()
    for entry in df["cuisine"].dropna()
    for token in str(entry).split(",")
    if token.strip()
))

cost_min = int(df["cost"].min())
cost_max = int(df["cost"].max())
rc_max   = int(df["rating_count"].max())


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍽️ Swiggy Restaurant Recommendation System")
st.caption(
    "Find the best restaurants tailored to your preferences. "
    "Set your filters in the sidebar and hit **Find Restaurants**."
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Your Preferences")

    city = st.selectbox("📍 City", all_cities)

    cuisine = st.selectbox("🍜 Cuisine", all_cuisines)

    min_rating = st.slider(
        "⭐ Minimum Rating",
        min_value=1.0, max_value=5.0, value=3.5, step=0.1
    )

    max_cost = st.slider(
        "💰 Max Cost for Two (₹)",
        min_value=cost_min, max_value=cost_max,
        value=min(500, cost_max), step=50
    )

    min_rc = st.slider(
        "👍 Min Number of Ratings",
        min_value=0, max_value=rc_max,
        value=20, step=10
    )

    method = st.radio(
        "🤖 Recommendation Method",
        options=["cosine", "kmeans"],
        format_func=lambda x: (
            "Cosine Similarity" if x == "cosine" else "K-Means Clustering"
        ),
    )

    top_n = st.slider("📋 Results to Show", 5, 20, 10)

    st.divider()
    find_btn = st.button("🚀 Find Restaurants", use_container_width=True)


# ── Main panel ────────────────────────────────────────────────────────────────
if find_btn:
    with st.spinner("🔍 Searching for the best matches…"):
        results = get_recommendations(
            city         = city,
            cuisine      = cuisine,
            rating       = min_rating,
            rating_count = float(min_rc),
            cost         = float(max_cost),
            method       = method,
            top_n        = top_n,
        )

    if results is None or results.empty:
        st.warning("😕 No recommendations found. Try relaxing your filters.")
        st.stop()

    st.success(f"✅ Found **{len(results)}** restaurant(s) for you!")

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏙️ City",           city)
    m2.metric("⭐ Avg Rating",      f"{results['rating'].mean():.2f}")
    m3.metric("💰 Avg Cost (₹)",    f"₹ {results['cost'].mean():.0f}")
    m4.metric("🎯 Avg Match",       f"{results['similarity_score'].mean():.0%}")

    st.divider()

    # ── Restaurant cards ──────────────────────────────────────────────────────
    st.subheader("🏆 Recommended Restaurants")

    for i, row in results.iterrows():
        with st.container():
            col_info, col_stats = st.columns([3, 1])

            with col_info:
                st.markdown(f"#### {i + 1}. {row['name']}")
                st.markdown(f"📍 **{row.get('city', '')}**")
                st.markdown(f"🍜 _{row.get('cuisine', 'N/A')}_")
                addr = row.get("address", "")
                if pd.notna(addr) and str(addr).strip() not in ("", "nan"):
                    st.caption(f"📌 {addr}")

            with col_stats:
                st.metric("Rating",  f"{row['rating']} ⭐")
                st.metric("Cost",    f"₹ {int(row['cost'])}")
                st.metric("Match",   f"{row['similarity_score']:.0%}")

            link = row.get("link", "")
            if pd.notna(link) and str(link).strip().startswith("http"):
                st.link_button("🔗 View on Swiggy", str(link))

            st.divider()

    # ── Full results table ────────────────────────────────────────────────────
    with st.expander("📊 View Full Results Table"):
        display_cols = [
            "name", "city", "cuisine", "rating",
            "rating_count", "cost", "similarity_score",
        ]
        show_cols = [c for c in display_cols if c in results.columns]
        st.dataframe(results[show_cols], use_container_width=True)

    # ── Download button ───────────────────────────────────────────────────────
    st.download_button(
        label     = "⬇️ Download Recommendations as CSV",
        data      = results.to_csv(index=False),
        file_name = "swiggy_recommendations.csv",
        mime      = "text/csv",
    )

else:
    # ── Welcome / dataset overview ────────────────────────────────────────────
    st.info("👈 Set your preferences in the sidebar and click **Find Restaurants** to begin.")

    with st.expander("📊 Dataset Overview", expanded=True):
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Total Restaurants", f"{len(df):,}")
        o2.metric("Cities",            f"{df['city'].nunique():,}")
        o3.metric("Cuisines",          f"{len(all_cuisines):,}")
        o4.metric("Avg Rating",        f"{df['rating'].mean():.2f} ⭐")

        st.markdown("**Sample Data (first 10 rows)**")
        st.dataframe(
            df[["name", "city", "cuisine", "rating", "rating_count", "cost"]].head(10),
            use_container_width=True,
        )
