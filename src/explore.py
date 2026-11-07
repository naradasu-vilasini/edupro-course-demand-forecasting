import pandas as pd

FILE = "data/EduPro_Online_Platform.xlsx"

sheets = pd.read_excel(FILE, sheet_name=None)
courses = sheets["Courses"]
transactions = sheets["Transactions"]

for name, table in sheets.items():
    print(name, table.shape[0], "rows,", table.shape[1], "columns")

print("\nMissing values:")
for name, table in sheets.items():
    print(name, int(table.isna().sum().sum()))

print("\nFree vs Paid courses:")
print(courses["CourseType"].value_counts())

per_course = transactions.groupby("CourseID").agg(
    enrollments=("TransactionID", "count"),
    revenue=("Amount", "sum"),
)
print("\nPer-course summary:")
print(per_course.describe().round(1))

merged = transactions.merge(courses, on="CourseID")
print("\nRevenue by category:")
print(merged.groupby("CourseCategory")["Amount"].sum().sort_values(ascending=False).round(0))