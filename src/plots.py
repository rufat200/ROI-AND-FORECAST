from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import shap

import config

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def _save_plot(fig, name, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / name
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Сохранено: {path}")


def plot_saturation_curve(results_df, output_dir="outputs"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in results_df["Model"].unique():
        subset = results_df[results_df["Model"] == model]
        subset = subset.sort_values("CPC ($)")
        ax.plot(subset["CPC ($)"], subset["ROI (%)"], marker="o", label=model)
        best_idx = subset["ROI (%)"].idxmax()
        best_cpc = subset.loc[best_idx, "CPC ($)"]
        best_roi = subset.loc[best_idx, "ROI (%)"]
        ax.scatter(best_cpc, best_roi)
        ax.annotate(f"opt={best_cpc}", (best_cpc, best_roi))
    ax.set_title("Saturation Curve (ROI vs CPC)")
    ax.legend()
    _save_plot(fig, "saturation_curve.png", output_dir)


def plot_incrementality(df, output_dir="outputs"):
    paid = df[df["medium"].str.contains("cpc|paid|ppc", case=False, na=False)]
    organic = df[~df["medium"].str.contains("cpc|paid|ppc", case=False, na=False)]
    paid_traffic = paid.groupby("event_hour")["page_views"].sum()
    organic_traffic = organic.groupby("event_hour")["page_views"].sum()
    aligned = pd.concat([paid_traffic, organic_traffic], axis=1).fillna(0)
    aligned.columns = ["paid", "organic"]
    aligned["incremental"] = aligned["paid"] - aligned["organic"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(aligned.index, aligned["incremental"])
    ax.set_title("Incremental Traffic (Ads Effect)")
    _save_plot(fig, "incrementality.png", output_dir)


def plot_grouped_importance(model, feature_names, output_dir="outputs"):
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })
    df["group"] = df["feature"].apply(lambda x: x.split("_")[0])
    grouped = df.groupby("group")["importance"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Grouped Feature Importance")
    _save_plot(fig, "grouped_importance.png", output_dir)


def plot_shap_importance(model, X, output_dir="outputs"):
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X, check_additivity=False)
        fig = plt.figure()
        shap.plots.bar(shap_values, show=False)
        _save_plot(fig, "shap_importance.png", output_dir)
    except Exception as e:
        print(f"[SHAP] ошибка: {e}")


def plot_decomposition(series, output_dir="outputs"):
    result = seasonal_decompose(series, period=24)
    fig = result.plot()
    _save_plot(fig, "decomposition.png", output_dir)


def plot_roi_vs_traffic(y_pred, df, cpc, conv_rate, output_dir="outputs"):
    users = df["users"].values
    ad_cost = cpc * users
    pred_conv = y_pred * conv_rate
    pred_revenue = pred_conv * config.AVG_ORDER_VALUE
    roi = np.where(ad_cost > 0, (pred_revenue - ad_cost) / ad_cost * 100, 0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, roi, alpha=0.5)
    ax.set_xlabel("Predicted Traffic")
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI vs Traffic")
    _save_plot(fig, "roi_vs_traffic.png", output_dir)


def plot_distribution(y_true, y_pred, output_dir="outputs"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(y_true, label="Real", ax=ax)
    sns.kdeplot(y_pred, label="Predicted", ax=ax)
    ax.legend()
    ax.set_title("Distribution Comparison")
    _save_plot(fig, "distribution.png", output_dir)


def plot_residuals(y_true, y_pred, output_dir="outputs"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    _save_plot(fig, "residuals.png", output_dir)


def plot_learning_curve(model, X, y, title="Learning Curve", output_dir="outputs"):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=3,
        scoring='neg_mean_absolute_error',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )
    train_scores = -train_scores.mean(axis=1)
    val_scores = -val_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores, label="Train Error")
    ax.plot(train_sizes, val_scores, label="Validation Error")
    ax.set_title(title)
    ax.set_xlabel("Train size")
    ax.set_ylabel("MAE")
    ax.legend()
    _save_plot(fig, f"learning_curve_{title}.png", output_dir)


def plot_metrics_comparison(results_df, output_dir="outputs"):
    """Сравнение sMAPE и других метрик между моделями."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x="CPC ($)", y="sMAPE (%)", hue="Model", ax=ax)
    ax.set_title("Сравнение точности моделей (sMAPE)")
    _save_plot(fig, "metrics_comparison.png", output_dir)


def plot_roi_sensitivity(results_df, output_dir="outputs"):
    df = results_df.copy()
    df["ROI_smooth"] = df.groupby("Model")["ROI (%)"] \
        .transform(lambda x: x.rolling(2, min_periods=1).mean())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="CPC ($)",
        y="ROI_smooth",
        hue="Model",
        marker="o",
        errorbar=None,
        estimator=None,
        ax=ax
    )
    ax.set_title("ROI Sensitivity (Smoothed)")
    _save_plot(fig, "roi_sensitivity.png", output_dir)


def plot_forecast_vs_actual(y_true, y_pred, title="Forecast", output_dir="outputs"):
    """Простой график Прогноз vs Факт для конкретной модели."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.values, label="Реальность", alpha=0.7)
    ax.plot(y_pred, label="Прогноз", linestyle="--")
    ax.set_title(title)
    ax.legend()
    _save_plot(fig, f"forecast_{title.replace(' ', '_')}.png", output_dir)


def plot_feature_importance(model, feature_names, output_dir="outputs"):
    """Важность признаков (работает для LightGBM и RF)."""
    if not hasattr(model, 'feature_importances_'):
        return
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_df, x='importance', y='feature', ax=ax, palette="viridis")
    ax.set_title(f"Top Features: {type(model).__name__}")
    _save_plot(fig, f"fi_{type(model).__name__}.png", output_dir)
