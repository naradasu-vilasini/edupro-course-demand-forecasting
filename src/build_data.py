import pandas as pd
from features import add_bucket_features

FILE = "data/EduPro_Online_Platform.xlsx"
sheets = pd.read_excel(FILE, sheet_name=None)
courses = sheets["Courses"]
teachers = sheets["Teachers"]
tx = sheets["Transactions"]

# 1) Month of every transaction
tx["Month"] = tx["TransactionDate"].dt.to_period("M").astype(str)

# 2) One row for every course and every month (60 courses x 12 months)
months = sorted(tx["Month"].unique())
grid = pd.MultiIndex.from_product(
    [courses["CourseID"], months], names=["CourseID", "Month"]
).to_frame(index=False)

# 3) Enrollments and revenue in each course-month (0 if nothing happened)
monthly = tx.groupby(["CourseID", "Month"]).agg(
    Enrollments=("TransactionID", "count"),
    Revenue=("Amount", "sum"),
).reset_index()
df = grid.merge(monthly, on=["CourseID", "Month"], how="left").fillna(0)
df = df.sort_values(["CourseID", "Month"]).reset_index(drop=True)

# 4) Course features
df = df.merge(courses.drop(columns=["CourseName"]), on="CourseID")

# 5) Teacher features for each course (average over the teachers who taught it)
t = tx.merge(teachers, on="TeacherID").merge(
    courses[["CourseID", "CourseCategory"]], on="CourseID")
t["ExpertiseMatch"] = (t["Expertise"] == t["CourseCategory"]).astype(int)
teacher_feat = t.groupby("CourseID").agg(
    AvgTeacherRating=("TeacherRating", "mean"),
    AvgTeacherExperience=("YearsOfExperience", "mean"),
    ExpertiseMatchScore=("ExpertiseMatch", "mean"),
).reset_index()
df = df.merge(teacher_feat, on="CourseID")

# 5b) Buckets: price band, duration bucket, rating tier, experience bucket, level code
df = add_bucket_features(df)

# 6) History features (only from EARLIER months, so no cheating)
g = df.groupby("CourseID")
df["PastEnrollments"] = g["Enrollments"].shift(1)
df["PastRevenue"] = g["Revenue"].shift(1)
df["PastEnrollAvg3"] = g["Enrollments"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df["PastRevenueAvg3"] = g["Revenue"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df["MonthNumber"] = df["Month"].str[-2:].astype(int)

# Revenue per enrollment in the previous month (0 when there were no enrollments)
df["PastRevPerEnroll"] = (df["PastRevenue"] / df["PastEnrollments"]).where(df["PastEnrollments"] > 0, 0)

# 7) The first month has no history, so remove it
df = df.dropna().reset_index(drop=True)

df.to_csv("data/model_table.csv", index=False)
print("Saved data/model_table.csv")
print("Rows:", df.shape[0], " Columns:", df.shape[1])
print(df.head(3).T)