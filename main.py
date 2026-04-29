import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import config
from src.features import build_feature_matrix
from src.data_split import split_data_ts
from src.models import (
    get_models,
    compute_traffic_metrics,
    compute_business_metrics
)
from src.model_cache import save_model, load_model
from src.plots import (
    plot_distribution,
    plot_forecast_vs_actual,
    plot_incrementality,
    plot_learning_curve,
    plot_metrics_comparison,
    plot_residuals,
    plot_roi_sensitivity,
    plot_roi_vs_traffic,
    plot_saturation_curve,
    plot_shap_importance,
)


warnings.filterwarnings("ignore")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "event_hour", "source", "medium",
        "device_type", "os",
        "users", "page_views", "conversions", "revenue"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    df["event_hour"] = pd.to_datetime(df["event_hour"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_hour"])
    numeric_cols = ["users", "page_views", "conversions", "revenue"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.sort_values("event_hour").reset_index(drop=True)
    print(f"[data] Загружено строк: {len(df):,}")
    return df


def run_sensitivity_analysis(df: pd.DataFrame):
    """Запуск обучения и оценки моделей для разных сегментов CPC."""
    all_results = []
    train_df, test_df = split_data_ts(df)
    for cpc in config.CPC_SEGMENTS:
        print(f"\n" + "═"*50)
        print(f" АНАЛИЗ ДЛЯ CPC = ${cpc:.2f}")
        print("═"*50)
        X_train, y_train, encoders = build_feature_matrix(train_df, cpc)
        X_test, y_test, _ = build_feature_matrix(test_df, cpc, encoders=encoders)
        models_dict = get_models()
        train_conv_rate = train_df["conversions"].sum() / train_df["users"].sum()
        for name, model in models_dict.items():
            print(f"[model] Обработка {name}...")
            if name == "XGBoost":
                model.fit(X_train, y_train)
            elif name == "LightGBM":
                import lightgbm as lgb
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(50)]
                )
            elif name == "CatBoost":
                model.fit(
                    X_train, y_train,
                    eval_set=(X_test, y_test),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            save_model(model, name, cpc)
            preds = model.predict(X_test)
            preds = np.maximum(preds, 0)
            t_metrics = compute_traffic_metrics(y_test, preds)
            is_paid_mask = test_df["medium"].str.lower().str.contains("cpc|paid|ppc", na=False).values
            b_metrics = compute_business_metrics(
                y_true=y_test.values,
                y_pred=preds,
                conversions=test_df["conversions"].values,
                revenue=test_df["revenue"].values,
                cpc=cpc,
                users=test_df["users"].values,
                is_paid_mask=is_paid_mask,
                conv_rate=train_conv_rate
            )
            res_row = {
                "CPC ($)": cpc,
                "Model": name,
                **t_metrics,
                **b_metrics
            }
            all_results.append(res_row)
            if cpc == 0.30:
                plot_forecast_vs_actual(y_test, preds, title=f"{name} CPC 0.30")
            if cpc == 0.30 and name == "LightGBM":
                plot_shap_importance(model, X_test)
                plot_residuals(y_test, preds)
                plot_distribution(y_test, preds)
                plot_learning_curve(model, X_train, y_train, title="LightGBM")
                plot_roi_vs_traffic(preds, test_df, cpc, train_conv_rate)
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="GA4 Traffic Forecasting Tool")
    parser.add_argument("--data", type=str, default=str(config.DATA_PATH), help="Путь к CSV")
    args = parser.parse_args()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        df = load_data(args.data)
        results_df = run_sensitivity_analysis(df)
        plot_incrementality(df)
        plot_saturation_curve(results_df)
        print("\n" + "─"*72)
        print(" ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (SENSITIVITY ANALYSIS)")
        print("─"*72)
        print(results_df.sort_values(["CPC ($)", "sMAPE (%)"]).to_string(index=False))
        results_df.to_csv(config.OUTPUT_DIR / "final_results.csv", index=False)
        plot_metrics_comparison(results_df)
        plot_roi_sensitivity(results_df)
        print(f"\n[success] Все отчеты и графики сохранены в {config.OUTPUT_DIR}")
    except Exception as e:
        print(f"[error] Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
