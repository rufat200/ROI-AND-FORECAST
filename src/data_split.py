import pandas as pd

import config


TEST_SIZE_RATIO = config.TEST_SIZE_RATIO
N_SPLITS = config.N_SPLITS
MIN_TRAIN_ROWS = config.MIN_TRAIN_ROWS

def split_data_ts(
    df: pd.DataFrame, ts_col: str="event_hour", test_ratio: float = TEST_SIZE_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"[split] Train: {train[ts_col].min()} -> {train[ts_col].max()} ({len(train)} строк)")
    print(f"[split] Test:  {test[ts_col].min()} -> {test[ts_col].max()} ({len(test)} строк)")
    return train, test


def get_walk_forward_indices(
    df: pd.DataFrame, 
    n_splits: int = N_SPLITS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = df.sort_values("event_hour").reset_index(drop=True)
    n = len(df)
    test_size = n // (n_splits + 1)
    train_base = n - n_splits * test_size

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for i in range(n_splits):
        train_end = train_base + i * test_size
        test_start = train_end
        test_end = test_start + test_size

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()
        if len(train) < MIN_TRAIN_ROWS:
            print(f"[wf] Фолд {i + 1} пропущен: train={len(train)} < {MIN_TRAIN_ROWS}")
            continue
        print(
            f"[wf] Fold {i+1}/{n_splits}\n"
            f"  train: rows={len(train):,} "
            f"[0:{train_end - 1}]\n"
            f"  test : rows={len(test):,} "
            f"[{test_start}:{test_end - 1}]"
        )
        folds.append((train, test))
        
    return folds
