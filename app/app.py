import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Find the files (works on your laptop and online) ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import add_bucket_features  # noqa: E402

DATA_FILE = os.path.join(ROOT, "data", "model_table.csv")
RESULTS_FILE = os.path.join(ROOT, "reports", "model_results.csv")
CATEGORY_FILE = os.path.join(ROOT, "reports", "category_forecast.csv")
FEATURES_FILE = os.path.join(ROOT, "reports", "features_used.json")

st.set_page_config(page_title="EduPro Demand & Revenue Forecast", layout="wide")

with open(FEATURES_FILE) as f:
    FEATURES = json.load(f)
CAT_COLS = FEATURES["cat_cols"]
NUM_COLS = FEATURES["num_cols"]


def make_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def train_model(target):
    df = load_data()
    # Same choice as in training: Lasso for enrollments, Ridge for revenue
    estimator = Lasso(alpha=0.1, max_iter=10000) if target == "Enrollments" else Ridge(alpha=1.0)
    model = Pipeline([("prep", make_prep()), ("model", estimator)])
    model.fit(df[CAT_COLS + NUM_COLS], df[target])
    return model


@st.cache_resource
def train_forest(target):
    df = load_data()
    model = Pipeline([("prep", make_prep()),
                      ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
    model.fit(df[CAT_COLS + NUM_COLS], df[target])
    return model


df = load_data()

st.title("EduPro: Course Demand and Revenue Forecasting")
st.caption("Predicts monthly enrollments and revenue for each course, using past EduPro data (2025).")

tab1, tab2, tab3, tab4 = st.tabs([
    "Course demand prediction",
    "Revenue forecast",
    "Feature importance",
    "Category comparison",
])

# ---------- Tab 1: predict for a course ----------
with tab1:
    st.subheader("Try a course")
    left, right = st.columns(2)
    with left:
        category = st.selectbox("Course category", sorted(df["CourseCategory"].unique()))
        level = st.selectbox("Course level", ["Beginner", "Intermediate", "Advanced"])
        ctype = st.selectbox("Course type", ["Paid", "Free"])
        price = st.slider("Course price", 0, 500, 250)
        if ctype == "Free":
            price = 0
        duration = st.slider("Duration (hours)", 1, 50, 25)
    with right:
        rating = st.slider("Course rating", 1.0, 5.0, 3.5, 0.1)
        t_rating = st.slider("Instructor rating", 3.5, 4.5, 4.1, 0.05)
        t_exp = st.slider("Average instructor experience (years)", 10.0, 20.0, 16.0, 0.5)
        match = st.slider("Instructor expertise matches category (0 to 1)", 0.0, 1.0, 0.2, 0.05)
        month = st.selectbox("Month to predict", list(range(1, 13)), index=0)

    typical_enroll = float(df["Enrollments"].mean())
    row = pd.DataFrame([{
        "CourseCategory": category, "CourseType": ctype, "CourseLevel": level,
        "CoursePrice": price, "CourseDuration": duration, "CourseRating": rating,
        "AvgTeacherRating": t_rating, "AvgTeacherExperience": t_exp,
        "ExpertiseMatchScore": match,
        "PastEnrollments": typical_enroll, "PastRevenue": typical_enroll * price,
        "PastEnrollAvg3": typical_enroll, "PastRevenueAvg3": typical_enroll * price,
        "PastRevPerEnroll": price, "MonthNumber": month,
    }])
    row = add_bucket_features(row)
    pred_enroll = max(0.0, float(train_model("Enrollments").predict(row[CAT_COLS + NUM_COLS])[0]))
    pred_rev = max(0.0, float(train_model("Revenue").predict(row[CAT_COLS + NUM_COLS])[0]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted enrollments (this month)", f"{pred_enroll:.0f}")
    c2.metric("Predicted revenue (this month)", f"{pred_rev:,.0f}")
    c3.metric("Price band / Rating tier", f"{row['PriceBand'][0]} / {row['RatingTier'][0]}")
    st.caption("Past enrollments are set to the platform average, since this is a new course idea.")

# ---------- Tab 2: revenue over time and forecast ----------
with tab2:
    st.subheader("Monthly revenue (past)")
    cats = st.multiselect("Choose categories", sorted(df["CourseCategory"].unique()),
                          default=["Artificial Intelligence", "Business"])
    sub = df[df["CourseCategory"].isin(cats)]
    monthly = sub.groupby(["Month", "CourseCategory"])["Revenue"].sum().reset_index()
    if monthly.empty:
        st.info("Choose at least one category.")
    else:
        st.plotly_chart(px.line(monthly, x="Month", y="Revenue", color="CourseCategory",
                                markers=True))

    st.subheader("Forecast check: predicted vs actual category revenue (Nov and Dec)")
    if os.path.exists(CATEGORY_FILE):
        cf = pd.read_csv(CATEGORY_FILE)
        long = cf.melt(id_vars=["Month", "CourseCategory"],
                       value_vars=["ActualRevenue", "PredictedRevenue"],
                       var_name="Type", value_name="Revenue")
        pick = st.selectbox("Category", sorted(cf["CourseCategory"].unique()))
        st.plotly_chart(px.bar(long[long["CourseCategory"] == pick], x="Month", y="Revenue",
                               color="Type", barmode="group"))
        st.dataframe(cf)
    else:
        st.info("Run src/train_models.py to create the category forecast.")

    st.subheader("How good are the models?")
    st.write("Learned from Feb to Oct, tested on Nov and Dec. R2 close to 1 is good; close to 0 is weak.")
    if os.path.exists(RESULTS_FILE):
        st.dataframe(pd.read_csv(RESULTS_FILE))
    else:
        st.info("Run src/train_models.py to create the results table.")

# ---------- Tab 3: what drives demand ----------
with tab3:
    st.subheader("What matters most?")
    target = st.selectbox("Predict", ["Enrollments", "Revenue"])
    forest = train_forest(target)
    names = forest.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(forest.named_steps["model"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False).head(10).reset_index()
    imp.columns = ["Feature", "Importance"]
    imp["Feature"] = imp["Feature"].str.replace("num__", "").str.replace("cat__", "")
    st.plotly_chart(px.bar(imp.iloc[::-1], x="Importance", y="Feature", orientation="h"))
    if target == "Revenue":
        st.write("Revenue follows price, because revenue is price times enrollments.")
    else:
        st.write("Enrollments are close to random in this data, so no single feature explains them strongly.")
    st.caption("Removed because they were too similar to course price: " + ", ".join(FEATURES["removed"]))

# ---------- Tab 4: compare categories ----------
with tab4:
    st.subheader("Category comparison")
    summary = df.groupby("CourseCategory").agg(
        Total_Enrollments=("Enrollments", "sum"),
        Total_Revenue=("Revenue", "sum"),
    ).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(summary.sort_values("Total_Enrollments"), x="Total_Enrollments",
                          y="CourseCategory", orientation="h", title="Enrollments by category"))
    b.plotly_chart(px.bar(summary.sort_values("Total_Revenue"), x="Total_Revenue",
                          y="CourseCategory", orientation="h", title="Revenue by category"))

    with st.expander("Data issues and assumptions"):
        st.write("- Only 60 courses exist, so the model table has one row per course per month (660 rows).")
        st.write("- Enrollments per course are very similar (140 to 196 in total), so they are hard to predict.")
        st.write("- Revenue is almost price times enrollments, and free courses earn 0.")
        st.write("- Many teachers teach each course, so teacher features are averages.")
        st.write("- The first month has no history and is removed.")
        st.write("- Category revenue is the sum of the predicted course revenue in that category.")