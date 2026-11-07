import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/model_table.csv")

# Inputs (features) the model can use
cat_cols = ["CourseCategory", "CourseType", "CourseLevel", "PriceBand",
            "DurationBucket", "RatingTier", "ExperienceBucket"]
num_cols = ["CoursePrice", "CourseDuration", "CourseRating", "LevelCode",
            "AvgTeacherRating", "AvgTeacherExperience", "ExpertiseMatchScore",
            "PastEnrollments", "PastRevenue", "PastEnrollAvg3", "PastRevenueAvg3",
            "PastRevPerEnroll", "MonthNumber"]

# Remove features that say almost the same thing (correlation above 0.9)
corr = df[num_cols].corr().abs()
dropped = []
for i, a in enumerate(num_cols):
    for b in num_cols[i + 1:]:
        if a not in dropped and b not in dropped and corr.loc[a, b] > 0.9:
            dropped.append(b)
            print(f"Removed {b} (too similar to {a}, correlation {corr.loc[a, b]:.2f})")
num_cols = [c for c in num_cols if c not in dropped]

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
with open("reports/features_used.json", "w") as f:
    json.dump({"cat_cols": cat_cols, "num_cols": num_cols, "removed": dropped}, f, indent=2)

# Time split: learn from Feb-Oct, test on Nov-Dec (the "future")
train = df[df["MonthNumber"] <= 10]
test = df[df["MonthNumber"] >= 11].copy()
print("\nTrain rows:", len(train), " Test rows:", len(test))

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}


def scores(y_true, y_pred):
    return (mean_absolute_error(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            r2_score(y_true, y_pred))


rows = []
for target, past_col in [("Enrollments", "PastEnrollments"), ("Revenue", "PastRevenue")]:
    y_train, y_test = train[target], test[target]

    # Simple baseline: "next month will be the same as last month"
    mae, rmse, r2 = scores(y_test, test[past_col])
    rows.append([target, "Baseline (last month)", mae, rmse, r2])

    best_name, best_r2, best_model, best_pred = None, -1e9, None, None
    for name, model in models.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(train[cat_cols + num_cols], y_train)
        pred = pipe.predict(test[cat_cols + num_cols])
        mae, rmse, r2 = scores(y_test, pred)
        rows.append([target, name, mae, rmse, r2])
        if r2 > best_r2:
            best_name, best_r2, best_model, best_pred = name, r2, pipe, pred

    joblib.dump(best_model, f"models/best_{target.lower()}.joblib")
    test["Predicted" + target] = np.maximum(best_pred, 0)
    print(f"Best model for {target}: {best_name}")

results = pd.DataFrame(rows, columns=["Target", "Model", "MAE", "RMSE", "R2"]).round(3)

# Third target: Category revenue = add up the predicted course revenue in each category
cat = test.groupby(["Month", "CourseCategory"]).agg(
    ActualRevenue=("Revenue", "sum"),
    PredictedRevenue=("PredictedRevenue", "sum"),
).reset_index()
mae, rmse, r2 = scores(cat["ActualRevenue"], cat["PredictedRevenue"])
cat_base = test.groupby(["Month", "CourseCategory"])["PastRevenue"].sum().values
bmae, brmse, br2 = scores(cat["ActualRevenue"], cat_base)
results.loc[len(results)] = ["Category Revenue", "Baseline (last month)", bmae, brmse, br2]
results.loc[len(results)] = ["Category Revenue", "Sum of course predictions", mae, rmse, r2]
results = results.round(3)
results.to_csv("reports/model_results.csv", index=False)

cat.round(2).to_csv("reports/category_forecast.csv", index=False)
test[["CourseID", "Month", "CourseCategory", "Enrollments", "PredictedEnrollments",
      "Revenue", "PredictedRevenue"]].round(2).to_csv("reports/test_predictions.csv", index=False)

print()
print(results.to_string(index=False))

# Which inputs matter most? (from a Random Forest on Enrollments)
rf = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
rf.fit(train[cat_cols + num_cols], train["Enrollments"])
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
imp = imp.sort_values(ascending=False).head(8).round(3)
imp.to_csv("reports/feature_importance.csv", header=["Importance"])
print("\nTop inputs for Enrollments:")
print(imp.to_string())