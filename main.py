import sys
import warnings
import argparse
from pathlib import Path

from time import perf_counter as p
 
import numpy as np
import pandas as pd
 
# Добавляем корень проекта в sys.path, чтобы импорты src.* работали
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "src")
 
from src import config
from src.init_data import load_data
from src.features import build_feature_matrix
from src.data_split import get_walk_forward_indices
from src.models import (
    get_models,
    compute_business_metrics,
    compute_traffic_metrics_log,
)
from src.model_cache import save_model, save_encoders
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
    plot_naive_comparison,
    plot_economic_interpretation,
)


warnings.filterwarnings("ignore")


def _fit_model(
    name: str,
    model, 
    X_train,
    y_train, 
    X_val, 
    y_val
) -> None:
    if name == "LightGBM":
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
    elif name == "CatBoost":
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
    elif name == "XGBoost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)


def run_sensitivity_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    all_results: list[dict] = []
    folds = get_walk_forward_indices(df)

    for cpc in config.CPC_SEGMENTS:
        print(f"\n{'='*50}\n CPC = ${cpc}\n{'='*50}")

        for name in get_models():
            print(f"[{name}] walk-forward обучение...")

            all_t_metrics: list[dict] = []
            all_b_metrics: list[dict] = []
            fold_smapes: list[dict] = []
            naive_smapes: list[dict] = []


            for i, (train_fold, test_fold) in enumerate(folds):
                start = p()

                X_train, y_train, encoders, seg_stats, _ = build_feature_matrix(train_fold, cpc)
                X_test, y_test, _, _, is_paid_mask = build_feature_matrix(test_fold, cpc, encoders, seg_stats)

                train_paid_mask = (
                    train_fold["medium"]
                    .str.lower()
                    .str.contains("cpc|paid|ppc", na=False)
                )
                paid_train = train_fold[train_paid_mask]
                if len(paid_train) > 0 and paid_train["users"].sum() > 0:
                    train_conv_rate = (
                        paid_train["conversions"].sum()
                        / paid_train["users"].sum()
                    )
                else:
                    train_conv_rate = (
                        train_fold["conversions"].sum()
                        / max(train_fold["users"].sum(), 1)
                    )
                    print(f"[warn] Fold {i+1}: нет платных строк в train, "
                          f"используется общий conv_rate={train_conv_rate:.6f}")


                y_naive = y_test.shift(1).bfill()
                naive_smapes.append(
                    compute_traffic_metrics_log(y_test, y_naive)["sMAPE (%)"]
                )

                model = get_models()[name]
                _fit_model(name, model, X_train, y_train, X_test, y_test)
                save_model(model, f"{name}_fold-{i+1}", cpc)
                save_encoders(encoders, seg_stats, name, i + 1, cpc)

                preds_log = np.maximum(model.predict(X_test), 0)
                preds = np.expm1(preds_log)


                t_m = compute_traffic_metrics_log(y_test, preds_log)
                b_m = compute_business_metrics(
                    y_true=np.expm1(y_test.values), 
                    y_pred=preds,
                    conversions=test_fold["conversions"].values,
                    revenue=test_fold["revenue"].values,
                    cpc=cpc, 
                    users=test_fold["users"].values,
                    is_paid_mask=is_paid_mask,
                    conv_rate=train_conv_rate,
                )
                all_t_metrics.append(t_m)
                all_b_metrics.append(b_m)
                fold_smapes.append(t_m["sMAPE (%)"])
                print(f"[{name}]")
                print(f"[Fold] {i+1} sMAPE = {t_m['sMAPE (%)']}%")
                print(f"[Fold] {i+1} R²    = {t_m['R²']*100:.2f}%")
                print(f"[Fold] {i+1} MAE   = {t_m['MAE']:.2f}")
                print(f"[Fold] {i+1} MedAE = {t_m['MedAE']:.2f}")
                print(f"[Fold] {i+1} MSE   = {t_m['MSE']:.2f}")
                print(f"[Fold] {i+1} RMSE  = {t_m['RMSE']:.2f}")

                is_last_fold = (i == len(folds) - 1)

                if cpc == 0.30 and is_last_fold:
                    y_test_origin = np.expm1(y_test.values)
                    plot_forecast_vs_actual(y_test_origin, preds, title=f"{name}_fold_{i+1}_CPC_0.30")
                    if name == "LightGBM":
                        plot_shap_importance(model, X_test)
                        plot_residuals(y_test, preds, title=f"residuals_{name}_fold_{i+1}_CPC_0.30")
                        plot_distribution(y_test, preds, title=f"distribution_{name}_fold_{i+1}_CPC_0.30")
                        plot_learning_curve(model_fn=lambda: get_models()["LightGBM"], 
                                            X_train=X_train, y_train=y_train, 
                                            X_val=X_test,
                                            y_val=y_test,
                                            title=f"{name}_fold_{i+1}_CPC_0.30")
                        plot_roi_vs_traffic(preds, test_fold, cpc, train_conv_rate, is_paid_mask, title=f"roi_vs_traffic_{name}_fold_{i+1}_CPC_0.30")
                
                print(f"[time] {p() - start: .2f}")

            avg_smape = float(np.mean(fold_smapes))
            avg_naive_smape = float(np.mean(naive_smapes))

            if avg_smape >= avg_naive_smape * 0.9:
                print(f"\n{'='*50}\n ПРЕДУПРЕЖДЕНИЕ: Модель {name} почти не лучше наивной!\n{'='*50}"
                      f"({avg_smape:.2f}%(fold) vs {avg_naive_smape:.2f}%(naive))\n")
            avg_t = pd.DataFrame(all_t_metrics).mean().to_dict()
            avg_b = pd.DataFrame(all_b_metrics).mean().to_dict()
            all_results.append({
                "CPC ($)": cpc, 
                "Model": name, 
                "sMAPE (%)": avg_smape,
                "Naive sMAPE (%)": avg_naive_smape,
                **avg_t,
                **avg_b,
            })

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.DATA_PATH))
    args = parser.parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    incr_df = plot_incrementality(df)
    paid_share = (
        incr_df["paid"].sum() /
        (incr_df["paid"].sum() + incr_df["organic"].sum()) * 100
    )
    print(f"[main] Доля платного трафика: {paid_share:.1f}%")

    results = run_sensitivity_analysis(df)

    plot_saturation_curve(results)
    plot_metrics_comparison(results)
    plot_roi_sensitivity(results)
    plot_economic_interpretation(results)
    plot_naive_comparison(results)
    plot_incrementality(df)

    print("\n" + "─"*72)
    print(results.sort_values(["CPC ($)", "sMAPE (%)"]).to_string(index=False))
    results.to_csv(config.OUTPUT_DIR / "final_results.csv", index=False)
    print(f"\n[success] Готово. Результаты в {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
