import pandas as pd


REQUIRED_COLS = {
    "event_hour", "source", "medium", "device_type", "os",
    "users", "page_views", "conversions", "revenue",
}

NUMERIC_COLS = ["users", "page_views", "conversions", "revenue"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)

    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    df["event_hour"] = pd.to_datetime(df["event_hour"], utc=True, errors="coerce")
    n_bad_ts = df["event_hour"].isna().sum()
    if n_bad_ts:
        print(f"[data] Удалено строк с невалидным event_hour: {n_bad_ts}")
    df = df.dropna(subset=["event_hour"])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    SEGMENT_KEYS = ["event_hour", "source", "medium", "device_type", "os"]
    before = len(df)
    df = df.groupby(SEGMENT_KEYS, as_index=False)[NUMERIC_COLS].sum()
    after = len(df)
    if before != after:
        print(f"[data] Агрегировано дубликатов: {before} → {after} строк")

    df = df.sort_values("event_hour").reset_index(drop=True)
    print(f"[data] Загружено строк: {len(df):,}")
    print(f"[data] Период: {df['event_hour'].min()} → {df['event_hour'].max()}")
    return df
