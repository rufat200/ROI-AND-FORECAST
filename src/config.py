from pathlib import Path


BASE_DIR = Path(__file__).parent # папка project/
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "bquxjob_3d66e4cb_19dc5bab84e.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── CPC-сетка для анализа чувствительности ───────────────────────────────────
CPC_SEGMENTS = [
    0.10, 
    0.20, 
    0.30, 
    0.40, 
    0.50, 
    0.60, 
    0.70,
]

TARGET_COL = "page_views"

TEST_SIZE_RATIO = 0.20 #20% тест
MIN_TRAIN_ROWS  = 50 
N_SPLITS = 5                     # количество фолдов


# ── Бизнес-константы ─────────────────────────────────────────────────────────
AVG_ORDER_VALUE = 50.0           # USD  – среднее revenue на conversion
AVG_CUSTOMER_LIFESPAN = 24       # месяцев
MONTHLY_CHURN_RATE = 0.05        # 5 % monthly churn assumed
GROSS_MARGIN = 0.40              # 40 % margin on revenue
CTR_DEFAULT = 0.03               # 3 % click-through rate (impressions → clicks)


USE_GPU = True


LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=100,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)

CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.03,
    depth=10,
    random_seed=42,
    verbose=0,
)


RF_PARAMS = dict(
    n_estimators=2000,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
