import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "month", "day_of_month",
    "week_of_year", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "ad_cost", "cpc_value",
    "lag_1", "lag_24", "rolling_mean_24",
    "source", "medium", "device_type", "os"
]

def extract_features(df, cpc, ts_col="event_hour"):
    df = df.copy()

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    dt = df[ts_col]

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    is_paid = df["medium"].str.lower().str.contains("cpc|paid|ppc", na=False)
    df["ad_cost"] = np.where(is_paid, cpc * df["users"].fillna(0), 0.0)
    df["cpc_value"] = cpc

    # lag features
    df["lag_1"] = df["page_views"].shift(1)
    df["lag_24"] = df["page_views"].shift(24)
    df["rolling_mean_24"] = df["page_views"].rolling(24).mean()

    df = df.dropna()

    return df


def build_feature_matrix(df, cpc, encoders=None, target_col="page_views"):
    df = extract_features(df, cpc)

    cat_cols = ["source", "medium", "device_type", "os"]

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in encoders:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("unknown")
            le.fit(df[col])
            encoders[col] = le

        le = encoders[col]
        df[col] = df[col].astype(str).fillna("unknown")

        # защита от unseen значений
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")

        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")

        df[col] = le.transform(df[col])

    X = df[FEATURE_COLS].copy()
    y = df[target_col].fillna(0)

    X = X.fillna(X.median(numeric_only=True))

    return X, y, encoders
