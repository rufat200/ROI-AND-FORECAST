from config import TEST_SIZE_RATIO, MIN_TRAIN_ROWS


def split_data_ts(df, ts_col="event_hour", test_ratio=TEST_SIZE_RATIO):
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)

    split_idx = int(n * (1 - test_ratio))
    if split_idx < MIN_TRAIN_ROWS:
        split_idx = min(MIN_TRAIN_ROWS, n - 1)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"[split] Train: {train[ts_col].min()} -> {train[ts_col].max()} ({len(train)} rows)")
    print(f"[split] Test:  {test[ts_col].min()} -> {test[ts_col].max()} ({len(test)} rows)")
    
    return train, test

def get_walk_forward_indices(df, n_splits=5):
    df = df.sort_values("event_hour").reset_index(drop=True)
    n = len(df)
    test_size = n // (n_splits + 1)
    
    folds = []
    for i in range(n_splits):
        train_end = n - (n_splits - i) * test_size
        test_end = train_end + test_size
        
        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()
        folds.append((train, test))
        
    return folds
