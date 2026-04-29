from pathlib import Path


BASE_DIR   = Path(__file__).parent
DATA_PATH = Path(r"E:\DATASETS\BigQuery\bquxjob_3d66e4cb_19dc5bab84e.csv")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CPC_SEGMENTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


TARGET_COL = "page_views"        # primary traffic forecast target


TEST_SIZE_RATIO = 0.20           # 20 % of data held out as test
MIN_TRAIN_ROWS  = 50             # guard rail for tiny datasets



AVG_ORDER_VALUE      = 50.0      # USD  – average revenue per conversion
AVG_CUSTOMER_LIFESPAN = 24       # months
MONTHLY_CHURN_RATE   = 0.05      # 5 % monthly churn assumed
GROSS_MARGIN         = 0.40      # 40 % margin on revenue
CTR_DEFAULT          = 0.03      # 3 % click-through rate (impressions → clicks)



LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

CATBOOST_PARAMS = dict(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)


RF_PARAMS = dict(
    n_estimators=600,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
