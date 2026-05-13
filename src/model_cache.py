import joblib
from pathlib import Path


CACHE_DIR = Path(__file__).parent.parent / "model_weights"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_path(model_name: str, cpc: float) -> Path: return CACHE_DIR / f"{model_name}_cpc_{cpc:.2f}.pkl"


def save_model(
    model, 
    model_name: str, 
    cpc: float,
) -> None:
    path = _get_path(model_name, cpc)
    joblib.dump(model, path)
    print(f"[cache] Сохранено: {path.name}")


def load_model(
    model_name: str, 
    cpc: float,
):
    path = _get_path(model_name, cpc)
    if path.exists():
        print(f"[cache] Загружено из кэша: {path.name}")
        return joblib.load(path)
    print(f"[cache] Кэш не найден: {path.name}")
    return None


def list_cached_models() -> list[dict]:
    results = []
    for p in sorted(CACHE_DIR.glob("*.pkl")):
        parts = p.stem.split("_cpc_")
        if len(parts) == 2:
            results.append({
                "model": parts[0],
                "cpc":   float(parts[1]),
                "path":  str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
    return results
