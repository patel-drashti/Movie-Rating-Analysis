"""
Movie Rating Analysis - Full Project Dashboard
PDS (Python for Data Science) Project

Covers:
- Part 1: Data loading, cleaning, EDA
- Part 2: SQLite database + SQL queries, ML rating prediction model
- Part 3: Interactive dashboard (this app)

Run with:  streamlit run app.py
"""

import os
import ssl
import zipfile
import sqlite3
import urllib.request
import certifi

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Rating Analysis",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
    html {
        color-scheme: light;
    }
    /* Hide Streamlit's default toolbar/menu/footer for a cleaner, website-like look */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Force a consistent light background everywhere, regardless of OS dark mode */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .block-container {
        background-color: #FFFFFF !important;
    }
    .block-container {
        padding-top: 2rem;
        color: #1A1A2E;
    }
    /* Ensure all body text is dark and readable on the white background */
    .block-container p,
    .block-container li,
    .block-container span,
    .block-container label,
    .block-container div {
        color: #1A1A2E;
    }

    .main-header {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 18px rgba(229,9,20,0.25);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.3rem;
    }
    .main-header p {
        color: #FFE5E5 !important;
        margin: 0.3rem 0 0 0;
        font-size: 1.05rem;
    }
    div[data-testid="stMetric"] {
        background-color: #F5F5F7 !important;
        border: 1px solid #E8E8ED !important;
        border-left: 5px solid #E50914 !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
    }
    div[data-testid="stMetric"] * {
        background-color: transparent !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1A1A2E !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #4A4A5A !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1A2E;
    }
    section[data-testid="stSidebar"] * {
        color: #F5F5F7 !important;
    }
    div[data-baseweb="radio"] label {
        padding: 0.35rem 0;
    }
    h1, h2, h3 {
        color: #1A1A2E !important;
    }
    /* Dataframes / tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #E8E8ED;
        border-radius: 8px;
    }
    /* Custom stat cards (replace st.metric, which fights browser dark mode) */
    .custom-metric {
        background-color: #F5F5F7 !important;
        border: 1px solid #E8E8ED;
        border-left: 5px solid #E50914;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .custom-metric-label {
        color: #4A4A5A !important;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .custom-metric-value {
        color: #1A1A2E !important;
        font-size: 1.9rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

def custom_metric(label, value):
    st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">{label}</div>
            <div class="custom-metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# Consistent color palette for all charts across the app
CHART_COLORWAY = ["#E50914", "#5B8FF9", "#5AD8A6", "#F6BD16", "#9270CA", "#E8684A", "#1A1A2E", "#5D7092"]
PLOTLY_TEMPLATE = "plotly_white"

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page background */
    .stApp {
        background-color: #0f1116;
    }

    /* Main content padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161a23;
        border-right: 1px solid #2a2f3a;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.02rem;
        padding: 4px 0;
    }

    /* Headings */
    h1 {
        color: #f5f6fa;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #e2e4ea;
        font-weight: 700;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1b1f2a 0%, #20263a 100%);
        border: 1px solid #2f3646;
        border-radius: 14px;
        padding: 18px 20px 12px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #9aa3b5 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #ff4b6e !important;
        font-weight: 800;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff4b6e !important;
        border-bottom-color: #ff4b6e !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2a2f3a;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff4b6e 0%, #ff7a59 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 75, 110, 0.35);
        color: white;
    }

    /* Success box after prediction */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        font-size: 1.1rem;
    }

    /* Sliders and multiselect accent */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #ff4b6e;
    }

    /* Horizontal divider */
    hr {
        border-color: #2a2f3a;
    }

    /* Caption text in sidebar */
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {
        color: #7d8494 !important;
    }
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "ml-latest-small"
ZIP_PATH = "ml-latest-small.zip"
DB_PATH = "movies.db"


# ----------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading & loading dataset...")
def load_data():
    if not os.path.exists(DATA_DIR):
        # Use certifi's trusted certificate bundle to avoid Windows SSL
        # "certificate verify failed" errors on fresh Python installs.
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATA_URL, context=ssl_context) as response:
            with open(ZIP_PATH, "wb") as out_file:
                out_file.write(response.read())
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype("float")
    df = pd.merge(ratings, movies, on="movieId")
    df["rating_year"] = pd.to_datetime(df["timestamp"], unit="s").dt.year

    return movies, ratings, df


@st.cache_resource(show_spinner="Building SQLite database...")
def build_database(movies, ratings):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    movies.to_sql("movies", conn, if_exists="replace", index=False)
    ratings.to_sql("ratings", conn, if_exists="replace", index=False)
    return conn


@st.cache_data(show_spinner="Training rating prediction model...")
def train_model(movies, df):
    # Build genre dummy features
    genre_dummies = movies["genres"].str.get_dummies("|")
    movies_feat = pd.concat([movies[["movieId", "year"]], genre_dummies], axis=1)

    # Aggregate rating stats per movie (target = average rating)
    movie_stats = df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    movie_stats.columns = ["movieId", "avg_rating", "num_ratings"]

    data = pd.merge(movies_feat, movie_stats, on="movieId").dropna()

    feature_cols = ["year", "num_ratings"] + list(genre_dummies.columns)
    X = data[feature_cols]
    y = data["avg_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    return model, feature_cols, list(genre_dummies.columns), rmse, mae


# ----------------------------------------------------------------------------
# Load everything
# ----------------------------------------------------------------------------
movies, ratings, df = load_data()
conn = build_database(movies, ratings)
model, feature_cols, genre_cols, rmse, mae = train_model(movies, df)

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 8px 0 20px 0;">
        <div style="font-size: 1.7rem; font-weight: 800; color: #f5f6fa;">🎬 Movie Rating</div>
        <div style="font-size: 1.7rem; font-weight: 800; color: #ff4b6e; margin-top: -8px;">Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA Dashboard", "Database (SQL)", "Prediction Model", "About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 Dataset: MovieLens (small)")
st.sidebar.caption(f"{len(movies):,} movies · {len(ratings):,} ratings")

# ----------------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    st.markdown("""
        <div class="main-header">
            <h1>🎬 Movie Rating Analysis</h1>
            <p>PDS Project — Full Pipeline: Data → Database → Model → Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        custom_metric("Total Movies", f"{len(movies):,}")
    with col2:
        custom_metric("Total Ratings", f"{len(ratings):,}")
    with col3:
        custom_metric("Unique Users", f"{ratings['userId'].nunique():,}")
    with col4:
        custom_metric("Avg Rating", f"{ratings['rating'].mean():.2f}")

    st.markdown("---")
    st.markdown("""
    ### Project Structure
    - **Part 1 — Data & EDA:** Cleaned and explored the MovieLens dataset (rating distribution, top movies, genre trends).
    - **Part 2 — Database & ML:** Stored data in a SQLite database with SQL queries, and trained a rating prediction model (Random Forest Regressor).
    - **Part 3 — Dashboard:** This interactive Streamlit app presenting all findings and the live prediction model.

    Use the sidebar to explore each section.
    """)

    st.markdown("### Sample of the data")
    st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: EDA Dashboard
# ----------------------------------------------------------------------------
elif page == "EDA Dashboard":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Exploratory Data Analysis</h1>
            <p>All charts from Part 1 — jump to any specific one below, or scroll through everything.</p>
        </div>
    """, unsafe_allow_html=True)

    movie_stats = df.groupby("title")["rating"].agg(["mean", "count"])
    movie_stats.columns = ["avg_rating", "num_ratings"]

    genre_dummies = movies["genres"].str.get_dummies("|")
    genre_counts = genre_dummies.sum().sort_values(ascending=False)
    genre_ratings = {}
    for genre in genre_dummies.columns:
        ids = movies.loc[genre_dummies[genre] == 1, "movieId"]
        genre_ratings[genre] = df[df["movieId"].isin(ids)]["rating"].mean()
    genre_rating_series = pd.Series(genre_ratings).sort_values(ascending=False)
    yearly_avg = df.groupby("rating_year")["rating"].mean()
    top_movies = movie_stats[movie_stats["num_ratings"] >= 50].sort_values(
        "avg_rating", ascending=False
    ).head(10)
    most_rated = movie_stats.sort_values("num_ratings", ascending=False).head(10)

    chart_options = [
        "Distribution of Ratings",
        "Top 10 Highest Rated Movies",
        "Top 10 Most Popular Movies",
        "Number of Movies per Genre",
        "Average Rating per Genre",
        "Ratings Trend Over the Years",
        "Correlation: Ratings vs Popularity",
        "Show All Charts",
    ]
    choice = st.selectbox("🔍 Jump to a specific chart", chart_options, index=len(chart_options) - 1)

    def show_ratings_dist():
        fig = px.histogram(df, x="rating", nbins=10, title="Distribution of Ratings",
                           color_discrete_sequence=[CHART_COLORWAY[0]], template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    def show_top_rated():
        fig = px.bar(top_movies, x="avg_rating", y=top_movies.index, orientation="h",
                    title="Top 10 Highest Rated Movies (min. 50 ratings)",
                    color_discrete_sequence=[CHART_COLORWAY[2]], template=PLOTLY_TEMPLATE)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    def show_most_popular():
        fig = px.bar(most_rated, x="num_ratings", y=most_rated.index, orientation="h",
                    title="Top 10 Most Popular Movies (by number of ratings)",
                    color_discrete_sequence=[CHART_COLORWAY[3]], template=PLOTLY_TEMPLATE)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    def show_genre_count():
        fig = px.bar(x=genre_counts.index, y=genre_counts.values,
                    title="Number of Movies per Genre",
                    labels={"x": "Genre", "y": "Count"},
                    color_discrete_sequence=[CHART_COLORWAY[1]], template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    def show_genre_rating():
        fig = px.bar(x=genre_rating_series.index, y=genre_rating_series.values,
                    title="Average Rating per Genre",
                    labels={"x": "Genre", "y": "Avg Rating"},
                    color_discrete_sequence=[CHART_COLORWAY[5]], template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    def show_yearly_trend():
        fig = px.line(x=yearly_avg.index, y=yearly_avg.values, markers=True,
                     title="Average Rating Given Per Year",
                     labels={"x": "Year", "y": "Average Rating"},
                     color_discrete_sequence=[CHART_COLORWAY[4]], template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    def show_correlation():
        fig = px.scatter(movie_stats, x="num_ratings", y="avg_rating", opacity=0.5,
                         title="Number of Ratings vs Average Rating",
                         color_discrete_sequence=[CHART_COLORWAY[6]], template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        corr = movie_stats["num_ratings"].corr(movie_stats["avg_rating"])
        custom_metric("Correlation Coefficient", f"{corr:.3f}")
        st.caption("A value near 0 = weak relationship, near 1 = strong positive relationship.")

    chart_map = {
        "Distribution of Ratings": show_ratings_dist,
        "Top 10 Highest Rated Movies": show_top_rated,
        "Top 10 Most Popular Movies": show_most_popular,
        "Number of Movies per Genre": show_genre_count,
        "Average Rating per Genre": show_genre_rating,
        "Ratings Trend Over the Years": show_yearly_trend,
        "Correlation: Ratings vs Popularity": show_correlation,
    }

    st.markdown("---")

    if choice == "Show All Charts":
        for name, func in chart_map.items():
            st.subheader(name)
            func()
            st.markdown("---")
    else:
        st.subheader(choice)
        chart_map[choice]()

# ----------------------------------------------------------------------------
# PAGE: Database (SQL)
# ----------------------------------------------------------------------------
elif page == "Database (SQL)":
    st.markdown("""
        <div class="main-header">
            <h1>🗄️ Database & SQL Queries</h1>
            <p>Part 2 — Data stored in SQLite, queried live with SQL.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("Data is stored in a **SQLite database** (`movies.db`) with two tables: `movies` and `ratings`.")

    queries = {
        "Top 10 rated movies (min 50 ratings)": """
            SELECT title, ROUND(AVG(rating),2) AS avg_rating, COUNT(*) AS num_ratings
            FROM ratings JOIN movies ON ratings.movieId = movies.movieId
            GROUP BY title
            HAVING num_ratings >= 50
            ORDER BY avg_rating DESC
            LIMIT 10;
        """,
        "Most active users": """
            SELECT userId, COUNT(*) AS num_ratings
            FROM ratings
            GROUP BY userId
            ORDER BY num_ratings DESC
            LIMIT 10;
        """,
        "Average rating per year of release": """
            SELECT movies.year, ROUND(AVG(ratings.rating),2) AS avg_rating, COUNT(*) AS num_ratings
            FROM ratings JOIN movies ON ratings.movieId = movies.movieId
            WHERE movies.year IS NOT NULL
            GROUP BY movies.year
            ORDER BY movies.year DESC
            LIMIT 15;
        """,
        "Number of movies per genre (SQL side)": """
            SELECT genres, COUNT(*) as count
            FROM movies
            GROUP BY genres
            ORDER BY count DESC
            LIMIT 10;
        """,
    }

    choice = st.selectbox("Choose a query to run", list(queries.keys()))
    st.code(queries[choice].strip(), language="sql")

    result = pd.read_sql(queries[choice], conn)
    st.dataframe(result, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Prediction Model
# ----------------------------------------------------------------------------
elif page == "Prediction Model":
    st.markdown("""
        <div class="main-header">
            <h1>🤖 Rating Prediction Model</h1>
            <p>Part 2 — Random Forest model predicting a movie's average rating.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    A **Random Forest Regressor** predicts a movie's average rating based on its
    **release year**, **estimated popularity (number of ratings)**, and **genres**.
    """)

    c1, c2 = st.columns(2)
    with c1:
        custom_metric("RMSE (Root Mean Squared Error)", f"{rmse:.3f}")
    with c2:
        custom_metric("MAE (Mean Absolute Error)", f"{mae:.3f}")
    st.caption("Lower values = better predictions. Ratings are on a 0.5–5.0 scale.")

    st.markdown("---")
    st.subheader("Try it yourself")

    col1, col2 = st.columns(2)
    with col1:
        year_input = st.slider("Release year", 1950, 2018, 2000)
        num_ratings_input = st.slider("Estimated number of ratings", 1, 300, 50)
    with col2:
        selected_genres = st.multiselect("Genres", genre_cols, default=["Comedy"])

    if st.button("Predict Rating"):
        input_row = {col: 0 for col in feature_cols}
        input_row["year"] = year_input
        input_row["num_ratings"] = num_ratings_input
        for g in selected_genres:
            if g in input_row:
                input_row[g] = 1

        input_df = pd.DataFrame([input_row])[feature_cols]
        prediction = model.predict(input_df)[0]

        st.success(f"⭐ Predicted average rating: **{prediction:.2f} / 5.0**")

# ----------------------------------------------------------------------------
# PAGE: About
# ----------------------------------------------------------------------------
elif page == "About":
    st.markdown("""
        <div class="main-header">
            <h1>ℹ️ About This Project</h1>
            <p>Tools, dataset, and pipeline overview.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    **Project:** Movie Rating Analysis
    **Subject:** PDS (Python for Data Science)

    **Dataset:** [MovieLens (small)](https://grouplens.org/datasets/movielens/) —
    ~100,000 ratings across ~9,700 movies by 600 users.

    **Tools used:**
    - Python, Pandas, NumPy — data cleaning & analysis
    - Plotly — interactive visualizations
    - SQLite — relational database & SQL queries
    - scikit-learn (Random Forest) — rating prediction model
    - Streamlit — this interactive dashboard

    **Pipeline:**
    1. Data collection & cleaning
    2. Exploratory Data Analysis (EDA)
    3. Database design & SQL querying
    4. Machine Learning model training & evaluation
    5. Interactive dashboard for presentation
    """)
