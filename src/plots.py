from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import learning_curve

import config

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]


# ── Вспомогательная: сохранение фигуры ───────────────────────────────────────
def _save(
    fig: plt.Figure, 
    name: str, 
    output_dir: str | Path = "outputs",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] Сохранено: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ЭКОНОМИКА 1
# ─────────────────────────────────────────────────────────────────────────────
def plot_economic_interpretation(
    results_df: pd.DataFrame, 
    output_dir: str | Path = "outputs",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=results_df, x="CPC ($)", y="Revenue ($)", hue="Model", ax=axes[0], marker="o")
    axes[0].set_title("Эффективность затрат (CPC vs Revenue)")
    sns.lineplot(data=results_df, x="CPC ($)", y="ROI (%)", hue="Model", ax=axes[1], marker="o")
    axes[1].set_title("Экономическая интерпретация (CPC vs ROI)")
    plt.tight_layout()
    _save(fig, "economic_interpretation.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ПРОГНОЗ vs РЕАЛЬНОСТЬ 2
# ─────────────────────────────────────────────────────────────────────────────
def plot_forecast_vs_actual(
    y_true,
    y_pred, 
    title: str = "Forecast", 
    output_dir: str | Path = "outputs",
) -> None:
    true_arr = np.array(y_true)
    pred_arr = np.array(y_pred)
    window = min(720, len(true_arr))
    t = true_arr[-window:]
    p = pred_arr[-window:]

    def smooth(arr: np.ndarray, k: int = 4) -> np.ndarray:
        return pd.Series(arr).rolling(k, min_periods=1).mean().values
    
    t_s = smooth(t)
    p_s = smooth(p)
    err = t_s - p_s

    denom = (np.abs(t_s) + np.abs(p_s)) / 2
    smape = float(np.mean(np.where(denom == 0, 0, np.abs(err) / denom ))) * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(len(t_s))
    axes[0].plot(x, t_s, label="Реальный трафик (GA4)", color="#2196F3", linewidth=1.5)
    axes[0].plot(x, p_s, label="Прогноз ИИ", color="#FF5722", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Сравнение прогноза и реальности: {title}  |  sMAPE={smape:.1f}%")
    axes[0].set_ylabel("Просмотры страниц")
    axes[0].legend()

    axes[1].bar(x, err, color=np.where(err >= 0, "#4CAF50", "#E53935"), alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Ошибка")
    axes[1].set_xlabel("Временные интервалы (часы)")

    plt.tight_layout()
    _save(fig, f"forecast_{title.replace(' ', '_')}.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# РАСПРЕДЕЛЕНИЕ 3
# ─────────────────────────────────────────────────────────────────────────────
def plot_distribution(
    y_true, 
    y_pred, 
    title: str = "distribution",
    output_dir: str | Path = "outputs",
) -> None:
    true_arr = np.array(y_true)
    pred_arr = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cap = np.percentile(true_arr, 99)
    t_c = np.clip(true_arr, 0, cap)
    p_c = np.clip(pred_arr, 0, cap)

    sns.kdeplot(t_c, label="Реальные", ax=axes[0], color="#2196F3", clip=(0, cap))
    sns.kdeplot(p_c, label="Спрогнозированные", ax=axes[0], color="#FF5722", clip=(0, cap))
    axes[0].set_title("Распределение (линейная шкала, без выбросов)")
    axes[0].set_xlabel("Просмотры страниц (page_views)")
    axes[0].legend()

    t_log = np.log1p(true_arr)
    p_log = np.log1p(pred_arr)
    sns.kdeplot(t_log, label="Реальные", ax=axes[1], color="#2196F3")
    sns.kdeplot(p_log, label="Спрогнозированные", ax=axes[1], color="#FF5722")
    axes[1].set_title("Распределение (log1p шкала)")
    axes[1].set_xlabel("log1p(page_views)")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, f"{title}.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ОСТАТКИ 4
# ─────────────────────────────────────────────────────────────────────────────
def plot_residuals(
    y_true,
    y_pred, 
    title: str = "residuals",
    output_dir: str | Path = "outputs",
) -> None:
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.2, s=4, color="#2196F3")
    axes[0].axhline(0, linestyle="--", color="red")
    axes[0].set_title("Анализ отклонений: Остатки vs Прогноз")
    axes[0].set_xlabel("Предсказанное значение")
    axes[0].set_ylabel("Остаток (true - pred)")

    sns.histplot(residuals, ax=axes[1], bins=50, kde=True, color="#2196F3")
    axes[1].axvline(float(np.mean(residuals)), color="red", linestyle="--", label=f"Bias={np.mean(residuals):.2f}")
    axes[1].set_title("Распределение остатков")
    axes[1].set_xlabel("Остаток")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, f"{title}.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP 5
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_importance(
    model, 
    X: pd.DataFrame, 
    output_dir: str | Path = "outputs",
    max_display: int = 20,
) -> None:
    try:
        X_sample = X.sample(min(2000, len(X)), random_state=42)
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample, check_additivity=False)

        shap.plots.bar(shap_values, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        _save(fig, "shap_importance.png", output_dir)
    except Exception as e:
        print(f"[SHAP] ошибка: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING CURVE 6
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curve(
    model, 
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    title: str = "Learning Curve", 
    output_dir: str | Path = "outputs",
) -> None:
    sizes, train_sc, val_sc = learning_curve(
        model, X, y, cv=3,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
    )
    train_mae = -train_sc.mean(axis=1)
    val_mae = -val_sc.mean(axis=1)
    train_std = train_sc.std(axis=1)
    val_std = val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sizes, val_mae, "o-", color="#FF9800", linewidth=2, label="Validation MAE")
    ax.plot(sizes, train_mae, "o-", color="#2196F3", linewidth=2, label="Train MAE")
    ax.fill_between(sizes, val_mae - val_std, val_mae + val_std, alpha=0.15, color="#FF9800")
    ax.fill_between(sizes, train_mae - train_std, train_mae + train_std, alpha=0.15, color="#2196F3")

    gap = val_mae - train_mae
    worst_idx = int(np.argmax(gap))
    ax.axvline(sizes[worst_idx], linestyle=":", color="red", alpha=0.7, label=f"Max gap @ {sizes[worst_idx]:,.0f} rows")
    ax.set_title(f"Learning Curve — LightGBM\n(Val ↓ и Train ↑ → сходятся = хорошо)")
    ax.set_xlabel("Размер Train")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save(fig, f"learning_curve_{title.replace(' ', '_')}.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ROI vs ТРАФИК 7
# ─────────────────────────────────────────────────────────────────────────────
def plot_roi_vs_traffic(
    y_pred: np.ndarray, 
    df: pd.DataFrame, 
    cpc: float, 
    conv_rate: float, 
    title: str = "roi_vs_traffic",
    output_dir: str | Path = "outputs",
) -> None:
    n = len(y_pred)
    if len(df) < n:
        raise ValueError(f"df ({len(df)}) короче y_pred ({n}) — выровняйте данные.")
    users = df["users"].iloc[:n].values
    ad_cost = cpc * users
    pred_rev = y_pred * conv_rate * config.AVG_ORDER_VALUE
    roi = np.where(ad_cost > 0, (pred_rev - ad_cost) / ad_cost * 100, np.nan)

    cap = np.nanpercentile(roi, 98)
    mask = np.isfinite(roi) & (roi <= cap) & (roi >= -100)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(y_pred[mask], roi[mask], alpha=0.3, s=5, c=roi[mask], cmap="RdYlGn", vmin=-50, vmax=cap)
    plt.colorbar(sc, ax=ax, label="ROI (%)")
    ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
    ax.set_title("ROI по отношению спрогнозированного трафика")
    ax.set_xlabel("Спрогнозированные просмотры страниц")
    ax.set_ylabel("ROI (%)")
    plt.tight_layout()
    _save(fig, f"{title}.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ИНКРЕМЕНТАЛЬНОСТЬ 8
# ─────────────────────────────────────────────────────────────────────────────
def plot_incrementality(
    df: pd.DataFrame, 
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    paid_mask = df["medium"].str.contains('cpc|paid|ppc', case=False, na=False)
    paid = df[paid_mask]
    organic = df[~paid_mask]

    paid_h = paid.groupby("event_hour")["page_views"].sum().rename("paid")
    org_h = organic.groupby("event_hour")["page_views"].sum().rename("organic")
    agg = pd.concat([paid_h, org_h], axis=1).fillna(0)
    agg["incremental"] = agg["paid"] - agg["organic"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})

    axes[0].plot(agg.index, agg["organic"], color="#FF9800", linewidth=0.8, label="Organic")
    axes[0].plot(agg.index, agg["paid"], color="#2196F3", linewidth=0.8, label="Paid")
    axes[0].set_title("Paid по отношению Organic трафики (по часовой)")
    axes[0].set_ylabel("Просмотр страниц (page_views)")
    axes[0].legend()

    pos = agg["incremental"].clip(lower=0)
    neg = agg["incremental"].clip(upper=0)
    axes[1].fill_between(agg.index, pos, 0, color="#4CAF50", alpha=0.7, label="Paid > Organic")
    axes[1].fill_between(agg.index, neg, 0, color="#E53935", alpha=0.7, label="Organic > Paid")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Инкрементальный трафик (Paid - Organic)")
    axes[1].set_ylabel("Δ Просмотр страниц (page_views)")
    axes[1].legend()

    total = (agg["paid"] + agg["organic"]).replace(0, np.nan)
    share = agg["paid"] / total * 100
    axes[2].fill_between(agg.index, share, alpha=0.6, color="#9C27B0")
    axes[2].set_title("Paid Traffic Share (%)")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    _save(fig, "incrementality.png", output_dir)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# КРИВАЯ НАСЫЩЕНИЯ 9
# ─────────────────────────────────────────────────────────────────────────────
def plot_saturation_curve(
    results_df: pd.DataFrame, 
    output_dir: str | Path = "outputs",
) -> None:
    models = results_df["Model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ylabel, title, ax in [
        ("sMAPE (%)", "sMAPE (%)", "Точность прогноза (sMAPE) vs CPC", axes[0]),
        ("R²", "R²", "Объяснённая дисперсия (R²) vs CPC", axes[1]),
    ]:
        for j, model in enumerate(models):
            sub = results_df[results_df["Model"] == model].sort_values("CPC ($)")
            ax.plot(sub["CPC ($)"], sub[metric], marker="o",
                    color=COLORS[j % len(COLORS)], label=model, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("CPC ($)")
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    _save(fig, "saturation_curve.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# СРАВНЕНИЕ МЕТРИК 10
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_comparison(
    results_df: pd.DataFrame, 
    output_dir: str | Path = "outputs",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=results_df, x="Model", y="sMAPE (%)", ax=axes[0], palette=COLORS, errorbar=None)
    axes[0].set_title("Средняя ошибка прогноза (ниже = лучше)")
    axes[0].set_ylabel("Ошибка (%)")

    sns.barplot(data=results_df, x="Model", y="R²", ax=axes[1], palette=COLORS, errorbar=None)
    axes[1].set_title("Коэффициент детерминации (выше = лучше)")
    axes[1].set_ylabel("Точность (R²)")

    plt.tight_layout()
    _save(fig, "model_comparison.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ROI-ЧУВСТВИТЕЛЬНОСТЬ 11
# ─────────────────────────────────────────────────────────────────────────────
def plot_roi_sensitivity(
    results_df: pd.DataFrame, 
    output_dir: str | Path = "outputs",
) -> None:
    models = results_df["Model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for j, model in enumerate(models):
        sub = results_df[results_df["Model"] == model].sort_values("CPC ($)")
        axes[0].plot(sub["CPC ($)"], sub["ROI (%)"], marker="o", label=model, linewidth=2, color=COLORS[j % len(COLORS)])
    axes[0].axhline(0, linestyle="--", color="black", linewidth=0.8, label="Break-even")
    axes[0].set_title("Зависимость окупаемости (ROI) от стоимости клика (CPC)")
    axes[0].set_xlabel("Стоимость клика (CPC ($))")
    axes[0].set_ylabel("Окупаемость (ROI (%))")
    axes[0].legend()

    ref = results_df[results_df["Model"] == models[0]].sort_values("CPC ($)")
    axes[1].plot(ref["CPC ($)"], ref["Ad Spend ($)"], marker="s", color="#E53935", label="Затраты на рекламу", linewidth=2)
    rev_col = "Pred Revenue ($)" if "Pred Revenue ($)" in ref.columns else "Revenue ($)"
    axes[1].plot(ref["CPC ($)"], ref[rev_col], marker="^", color="#4CAF50", label="Ожидаемая выручка", linewidth=2)
    
    axes[1].set_title("Бизнес-показатели")
    axes[1].set_xlabel("CPC ($)")
    axes[1].set_ylabel("Сумма ($)")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, "business_efficiency.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Naive baseline vs модели 12
# ─────────────────────────────────────────────────────────────────────────────
def plot_naive_comparison(
    results_df: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> None:
    if "Naive sMAPE (%)" not in results_df.columns:
        print("[plot] plot_naive_comparison: колонка 'Naive sMAPE (%)' отсутствует.")
        return
 
    df_agg = (
        results_df.groupby("Model")[["sMAPE (%)", "Naive sMAPE (%)"]]
        .mean()
        .reset_index()
        .sort_values("sMAPE (%)")
    )
    df_agg["Improvement (pp)"] = df_agg["Naive sMAPE (%)"] - df_agg["sMAPE (%)"]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    x = np.arange(len(df_agg))
    w = 0.35
    axes[0].bar(x - w/2, df_agg["sMAPE (%)"],       w, label="Model",  color="#2196F3")
    axes[0].bar(x + w/2, df_agg["Naive sMAPE (%)"], w, label="Naive",  color="#FF9800")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_agg["Model"])
    axes[0].set_title("sMAPE: Модель vs Наивный прогноз (ниже = лучше)")
    axes[0].set_ylabel("sMAPE (%)")
    axes[0].legend()
 
    colors = ["#4CAF50" if v >= 0 else "#E53935" for v in df_agg["Improvement (pp)"]]
    axes[1].bar(df_agg["Model"], df_agg["Improvement (pp)"], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Выигрыш над наивным прогнозом (pp sMAPE)")
    axes[1].set_ylabel("Δ sMAPE (pp)")
 
    plt.tight_layout()
    _save(fig, "naive_comparison.png", output_dir)
