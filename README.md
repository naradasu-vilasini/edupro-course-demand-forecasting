# EduPro: Course Demand and Revenue Forecasting

Machine Learning project for the Unified Mentor internship.
It predicts monthly course enrollments and revenue for the EduPro online learning platform (2025 data) and shows the results in a Streamlit dashboard.

## Run
    pip install -r requirements.txt
    python src/build_data.py
    python src/train_models.py
    streamlit run app/app.py

## Structure
- data/EduPro_Online_Platform.xlsx   original data (Users, Teachers, Courses, Transactions)
- data/model_table.csv               one row per course per month (660 rows)
- src/features.py                    price bands, duration buckets, rating tiers, experience buckets
- src/build_data.py                  merges the sheets and builds the model table
- src/train_models.py                trains and compares the models, saves results to reports/
- app/app.py                         Streamlit dashboard
- reports/                           model results, category forecast, feature list

## Targets
- Enrollment count per course (per month)
- Course revenue (per month)
- Category revenue (sum of predicted course revenue per category)

## Features
- Course: price band, duration bucket, rating tier, level code, type, category
- Instructor: experience bucket, average rating, expertise-category match score
- History: past enrollments, past revenue, 3-month averages, revenue per enrollment
- Removed because they were too similar to course price (correlation above 0.9): PastRevenue, PastRevenueAvg3, PastRevPerEnroll

## Models
Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting.
Learned from Feb to Oct 2025 and tested on Nov and Dec 2025.

| Target | Best model | MAE | R2 |
|---|---|---|---|
| Enrollments | Lasso | 2.76 | 0.006 |
| Revenue | Ridge | 288.1 | 0.933 |
| Category revenue | Sum of course predictions | 925.0 | 0.939 |

## Data issues and assumptions
- Only 60 courses exist, so the model table uses one row per course per month.
- Enrollments per course are very similar (140 to 196 in total), so they are hard to predict. The models are not better than a simple average.
- Revenue is almost price times enrollments, and free courses earn 0, so revenue is easier to predict.
- Many teachers teach each course, so instructor features are averages.
- The first month has no history and is removed.
- Category revenue is the sum of the predicted course revenue in that category.