import joblib
from pathlib import Path


CACHE_DIR = Path(__file__).parent.parent / "model_weights"
CACHE_DIR.mkdir(exist_ok=True)

def get_path(model_name: str, cpc: float) -> Path:
    return CACHE_DIR / f"{model_name}_cpc_{cpc:.2f}.pkl"

def save_model(model, model_name: str, cpc: float) -> None:
    path = get_path(model_name, cpc)
    joblib.dump(model, path)
    print(f"[cache] Перезаписано: {path.name}")

def load_model(model_name, cpc):
    path = get_path(model_name, cpc)
    return joblib.load(path) if path.exists() else None
