import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    # Временные
    "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "hour_x_dow",
    "is_holiday_season", # ноябрь-декабрь = 1
    "days_since_jan1", # позиция в году, циклические провалы
    "page_views_per_user",
    # Рекламные
    "ad_cost", "cpc_value", "adstock_cost", "ad_cost_lag_1", "ad_cost_lag_24",
    # Лаговые (почасовые агрегаты)
    "lag_1", "lag_4", "lag_8", "lag_12", "lag_24", "lag_48", "lag_168",
    # Скользящие / тренд
    "rolling_mean_24", "rolling_std_24", "slope_24",
    "rolling_mean_48", "rolling_mean_168",
    # Сегментные средние (target-encoding без утечки)
    "seg_mean_users", "seg_std_users", "seg_ratio", 
    "seg_lag_1", 
    "seg_lag_24",
    # Категориальные
    "source", "medium", "device_type", "os",
]

CAT_COLS = ["source", "medium", "device_type", "os"]
SEG_COLS = ["source", "medium", "device_type", "os"]


def _rolling_slope(series: pd.Series, window: int = 24):
    def _slope(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        x -= x.mean()
        denom = (x ** 2).sum()
        if denom == 0:
            return 0.0
        slope = (x * arr).sum() / denom
        scale = np.abs(arr).mean() or 1.0
        return float(np.clip(slope / scale, -1.0, 1.0))
    return series.rolling(window, min_periods=2).apply(_slope, raw=True)


def _adstock(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = x[i] + alpha * result[i - 1]
    return result

def compute_segment_stats(
    df: pd.DataFrame, seg_cols: list[str], value_col: str = "users", train_stats: dict | None = None
) -> tuple[pd.Series, pd.Series, dict]:
    seg_key = df[seg_cols].astype(str).apply(lambda r: "|".join(r), axis=1)
    original_index = df.index

    if train_stats is None:
        df_local = df.reset_index(drop=True).copy()
        df_local["_seg"] = seg_key.values
        df_local["_val"] = df_local[value_col].values
        means: list[float] = []
        stds: list[float] = []
        seg_expanding: dict[str, list[float]] = {}

        for seg, val in zip(df_local["_seg"], df_local["_val"]):
            hist = seg_expanding.setdefault(seg, [])
            if hist:
                means.append(float(np.mean(hist)))
                stds.append(float(np.std(hist)) if len(hist) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)
            hist.append(float(val))

        train_stats = {
            seg: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
            for seg, vals in seg_expanding.items()
        }

        global_mean = float(df_local[value_col].mean())
        global_std = float(df_local[value_col].std())

        seg_mean = pd.Series(means, index=original_index).fillna(global_mean)
        seg_std = pd.Series(stds, index=original_index).fillna(global_std)
    else:
        global_mean = float(np.mean([v["mean"] for v in train_stats.values()]))
        global_std = float(np.mean([v["std"] for v in train_stats.values()]))

        seg_mean = seg_key.map(lambda s: train_stats.get(s, {}).get("mean", global_mean)).astype(float)
        seg_std = seg_key.map(lambda s: train_stats.get(s, {}).get("std",  global_std)).astype(float)
    return seg_mean, seg_std, train_stats


def _compute_segment_lags(
    df: pd.DataFrame, seg_cols: list[str], target_col: str = "page_views", lags: list[int] = [1, 24],
) -> pd.DataFrame:
    seg_key = df[seg_cols].astype(str).apply(lambda r: "|".join(r), axis=1)
    df = df.copy()
    df["_seg"] = seg_key
 
    result = pd.DataFrame(index=df.index)
    for lag in lags:
        col_name = f"seg_lag_{lag}"
        result[col_name] = df.groupby("_seg")[target_col].shift(lag)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Основная функция построения матрицы признаков
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(
    df: pd.DataFrame,
    cpc: float,
    encoders: dict | None = None,
    seg_stats: dict | None = None,
    target_col: str = "page_views",
    log_transform: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict, dict, np.ndarray]:
    """
    Возвращает (X, y, encoders, seg_stats, is_paid_mask).
    """
    df = df.copy().sort_values("event_hour").reset_index(drop=True)
    dt = pd.to_datetime(df["event_hour"], utc=True)
 
    is_train = seg_stats is None
 
    # ── 1. Временные признаки ─────────────────────────────────────────────────
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_x_dow"] = df["hour"] * 7 + df["day_of_week"]

    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    day_of_year = dt.dt.dayofyear
    df["days_since_jan1"] = day_of_year / 365.0

    seg_pv_per_user = df.groupby("event_hour").apply(lambda g: (g["page_views"] / g["users"].clip(lower=1)).mean()).shift(1)
    df["page_views_per_user"] = df["event_hour"].map(seg_pv_per_user).fillna((df["page_views"] / df["users"].clip(lower=1)).median())
 
    # ── 2. Рекламные признаки (БЕЗ утечки таргета) ───────────────────────────
    is_paid = df["medium"].str.lower().str.contains("cpc|paid|ppc", na=False)
    is_paid_mask = is_paid.values

    df["ad_cost"] = np.where(is_paid, cpc * df["users"], 0.0)
    df["cpc_value"] = cpc
    df["ad_cost_lag_1"] = df["ad_cost"].shift(1).fillna(0.0)
    df["ad_cost_lag_24"] = df["ad_cost"].shift(24).fillna(0.0)
 
    # ── 3. Почасовые лаги по агрегированному ряду ────────────────────────────

    hourly_log = df.groupby("event_hour")["page_views"].sum().apply(np.log1p).sort_index()
 
    for lag in [1, 4, 8, 12, 24, 48, 168]:
        shifted = hourly_log.shift(lag)
        df[f"lag_{lag}"] = df["event_hour"].map(shifted)
 
    hourly_roll_mean_24 = hourly_log.shift(1).rolling(24, min_periods=1).mean()
    hourly_roll_mean_48 = hourly_log.shift(1).rolling(48, min_periods=1).mean()
    hourly_roll_mean_168 = hourly_log.shift(1).rolling(168, min_periods=1).mean()
    hourly_roll_std = hourly_log.shift(1).rolling(24, min_periods=1).std().fillna(0)
    hourly_slope = _rolling_slope(hourly_log.shift(1).bfill(), window=24)
 
    df["rolling_mean_24"] = df["event_hour"].map(hourly_roll_mean_24)
    df["rolling_mean_48"] = df["event_hour"].map(hourly_roll_mean_48)
    df["rolling_mean_168"] = df["event_hour"].map(hourly_roll_mean_168)
    df["rolling_std_24"] = df["event_hour"].map(hourly_roll_std)
    df["slope_24"] = df["event_hour"].map(hourly_slope)


    lag_cols = [f"lag_{l}" for l in [1, 4, 8, 12, 24, 48, 168]] + \
               ["rolling_mean_24", "rolling_mean_48", "rolling_mean_168", "rolling_std_24", "slope_24"]
 
    if encoders is None:
        encoders = {}
    if is_train:
        lag_medians = {col: df[col].median() for col in lag_cols}
        encoders["lag_medians"] = lag_medians
    else:
        lag_medians = encoders.get("lag_medians", {})
    for col in lag_cols:
        fill_val = lag_medians.get(col, df[col].median())
        df[col] = df[col].fillna(fill_val)
 
    # ── 4. Сегментные средние без утечки ─────────────────────────────────────
    seg_mean, seg_std, seg_stats = compute_segment_stats(df, SEG_COLS, value_col="users", train_stats=seg_stats)
    df["seg_mean_users"] = seg_mean.values
    df["seg_std_users"]  = seg_std.values
    df["seg_ratio"] = df["users"] / (seg_mean.values.clip(min=1e-9))
    
    seg_lags_df = _compute_segment_lags(df, SEG_COLS, target_col, lags=[1, 24])
    df["seg_lag_1"]  = seg_lags_df["seg_lag_1"].fillna(seg_lags_df["seg_lag_1"].median())
    df["seg_lag_24"] = seg_lags_df["seg_lag_24"].fillna(seg_lags_df["seg_lag_24"].median())
 
    # ── 5. Adstock (ИСПРАВЛЕН: первый элемент теперь = x[0]) ─────────────────
    df["adstock_cost"] = _adstock(df["ad_cost"].values)
 
    # ── 6. Label encoding ─────────────────────────────────────────────────────
    for col in CAT_COLS:
        df[col] = df[col].astype(str).fillna("unknown")
        if col not in encoders:
            le = LabelEncoder()
            all_vals = list(df[col].unique()) + ["unknown"]
            le.fit(all_vals)
            encoders[col] = le
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "unknown")
        df[col] = le.transform(df[col])
 
    # ── 7. Сборка X, y ────────────────────────────────────────────────────────
    X = df[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    if log_transform:
        y = np.log1p(df[target_col].fillna(0))
    else:
        y = df[target_col].fillna(0)

    return X, y, encoders, seg_stats, is_paid_mask
