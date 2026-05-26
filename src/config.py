from pathlib import Path


BASE_DIR = Path(__file__).parent # папка project/
DATA_PATH = Path(r"C:\Users\Rufat\Downloads\Telegram Desktop\bquxjob_3d66e4cb_19dc5bab84e.csv")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── CPC-сетка для анализа чувствительности ───────────────────────────────────
CPC_SEGMENTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# ── Целевая переменная ────────────────────────────────────────────────────────
TARGET_COL = "page_views"

# ── Разбивка данных ───────────────────────────────────────────────────────────
TEST_SIZE_RATIO = 0.20           # 20 % данных идут на тест
MIN_TRAIN_ROWS  = 50             # нужно если в датасете (в фолде) СЛИШКОМ мало данных
N_SPLITS = 5                     # количество фолдов


# ── Бизнес-константы ─────────────────────────────────────────────────────────
AVG_ORDER_VALUE = 50.0           # USD  – среднее revenue на conversion
AVG_CUSTOMER_LIFESPAN = 24       # месяцев
MONTHLY_CHURN_RATE = 0.05        # 5 % monthly churn assumed
GROSS_MARGIN = 0.40              # 40 % margin on revenue
CTR_DEFAULT = 0.03               # 3 % click-through rate (impressions → clicks)


USE_GPU = False


LGBM_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.09,
    num_leaves=30,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


XGB_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.09,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)

CATBOOST_PARAMS = dict(
    iterations=150,
    learning_rate=0.09,
    depth=3,
    random_seed=42,
    verbose=0,
)


RF_PARAMS = dict(
    n_estimators=150,
    max_depth=3,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
