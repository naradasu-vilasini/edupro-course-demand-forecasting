import pandas as pd


def add_bucket_features(df):
    """Adds price bands, duration buckets, rating tiers, experience buckets and level code."""
    df = df.copy()
    df["PriceBand"] = pd.cut(df["CoursePrice"], [-0.01, 0, 170, 330, 100000],
                             labels=["Free", "Low", "Medium", "High"]).astype(str)
    df["DurationBucket"] = pd.cut(df["CourseDuration"], [0, 15, 35, 1000],
                                  labels=["Short", "Medium", "Long"]).astype(str)
    df["RatingTier"] = pd.cut(df["CourseRating"], [0, 2.5, 4, 10], right=False,
                              labels=["Low", "Medium", "High"]).astype(str)
    df["ExperienceBucket"] = pd.cut(df["AvgTeacherExperience"], [0, 15.5, 16.5, 1000],
                                    right=False, labels=["Low", "Medium", "High"]).astype(str)
    df["LevelCode"] = df["CourseLevel"].map({"Beginner": 0, "Intermediate": 1, "Advanced": 2})
    return df