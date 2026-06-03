"""
predict_cli.py — консольный инструмент прогноза трафика и оценки рекламы.

Запуск:
    uv run predict_cli.py
    uv run predict_cli.py --date 2026-05-25 --source google --medium cpc --cpc 0.30
    uv run predict_cli.py --help
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "src")

import joblib
from src import config
from src.features import build_feature_matrix
from src.init_data import load_data
from src.model_cache import load_encoders


# ── Константы отображения ─────────────────────────────────────────────────────
W = 64

def _line(char="─"): print(char * W)
def _header(text): _line("═"); print(f"  {text}"); _line("═")
def _section(text): print(); _line(); print(f"  {text}"); _line()

def _bar(value, max_val, width=30, char="█", empty="░"):
    filled = round(value / max_val * width) if max_val > 0 else 0
    return char * filled + empty * (width - filled)

def _color(text, code): return f"\033[{code}m{text}\033[0m"

GREEN = lambda t: _color(t, "32")
RED = lambda t: _color(t, "31")
YELLOW = lambda t: _color(t, "33")
CYAN = lambda t: _color(t, "36")
BOLD = lambda t: _color(t, "1")


# ── Справочники ───────────────────────────────────────────────────────────────
SOURCES = ["google", "(direct)", "<Other>", "shop.googlemerchandisestore.com", "(data deleted)"]
MEDIUMS = ["organic", "(none)", "cpc", "referral", "<Other>", "(data deleted)"]
DEVICES = ["desktop", "mobile", "tablet"]
OS_LIST = ["Windows", "Web", "Android", "iOS", "Macintosh", "<Other>"]
MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
FOLDS = [str(i) for i in range(1, config.N_SPLITS + 1)]
CPCS = list(config.CPC_SEGMENTS)


# ── Вспомогательные функции ───────────────────────────────────────────────────
def _ask(prompt, options=None, default=None, cast=str):
    while True:
        if options:
            for i, o in enumerate(options, 1):
                print(f"    {i}. {o}")
            hint = f"[1-{len(options)}]" + (f", Enter={default}" if default is not None else "")
            raw = input(f"  {prompt} {hint}: ").strip()
            if raw == "" and default is not None:
                return default
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            print(RED(f"  Введите число от 1 до {len(options)}."))
        else:
            hint = f"(Enter={default})" if default is not None else ""
            raw = input(f"  {prompt} {hint}: ").strip()
            if raw == "" and default is not None:
                return default
            try:
                return cast(raw)
            except (ValueError, TypeError):
                print(RED("  Неверный формат. Попробуйте снова."))


def load_model_weights(model_name: str, cpc: float, fold):
    fold = int(fold)
    ch = f"./model_weights/{model_name}_fold-{fold}_cpc_{cpc:.2f}.pkl"
    try:
        return joblib.load(ch)
    except Exception as e:
        print(RED(f"  Ошибка загрузки {ch}:\n  {e}"))
        return None


def compute_conv_rate(history_df: pd.DataFrame) -> float:
    paid_mask = history_df["medium"].str.lower().str.contains("cpc|paid|ppc", na=False)
    paid = history_df[paid_mask]
    total_users = paid["users"].sum()
    if total_users == 0:
        return 0.015
    return float(paid["conversions"].sum() / total_users)

def build_prediction_context(
    params: dict,
    history_df: pd.DataFrame,
    context_weeks: int = 2,
) -> tuple[pd.DataFrame, int, list[int], dict]:
    date_str = params["date"]
    pred_date = pd.Timestamp(date_str, tz="UTC")

    # ── 1. Исторический паттерн сегмента ─────────────────────────────────────
    seg_mask = (
        (history_df["source"] == params["source"]) &
        (history_df["medium"] == params["medium"]) &
        (history_df["device_type"] == params["device"]) &
        (history_df["os"] == params["os"])
    )
    seg_hist = history_df[seg_mask].copy()

    if len(seg_hist) < 48:
        # Нет данных по сегменту — берём глобальный контекст
        print("[warn] Мало данных для сегмента, использую глобальный паттерн.")
        seg_hist = history_df.copy()

    dt = pd.to_datetime(seg_hist["event_hour"])
    seg_hist["_hour"] = dt.dt.hour
    seg_hist["_dow"]  = dt.dt.dayofweek

    # Средние users и page_views по (day_of_week, hour)
    pattern = (
        seg_hist
        .groupby(["_dow", "_hour"])
        .agg(users_mean=("users", "mean"), pv_mean=("page_views", "mean"))
        .reset_index()
    )
    global_u = float(seg_hist["users"].mean()) or 1.0
    global_pv = float(seg_hist["page_views"].mean()) or 1.0

    def _lookup(dow: int, hour: int) -> tuple[int, int]:
        row = pattern[(pattern["_dow"] == dow) & (pattern["_hour"] == hour)]
        if len(row):
            u = max(1, round(float(row["users_mean"].values[0])))
            pv = max(0, round(float(row["pv_mean"].values[0])))
        else:
            u = max(1, round(global_u))
            pv = max(0, round(global_pv))
        return u, pv

    # ── 2. Синтетическая история: context_weeks недель до прогноза ───────────
    ctx_start = pred_date - pd.Timedelta(hours=context_weeks * 7 * 24)
    ctx_hours = pd.date_range(
        start=ctx_start,
        end=pred_date - pd.Timedelta(hours=1),
        freq="h", tz="UTC",
    )
    ctx_rows = []
    for ts in ctx_hours:
        u, pv = _lookup(ts.dayofweek, ts.hour)
        ctx_rows.append({
            "event_hour": ts, "source": params["source"],
            "medium": params["medium"], "device_type": params["device"],
            "os": params["os"], "users": u, "page_views": pv,
            "conversions": 0, "revenue": 0.0,
        })
    context_df = pd.DataFrame(ctx_rows)

    # ── 3. Окно прогноза: 24 часа в нужный день ──────────────────────────────
    pred_hours = pd.date_range(
        start=f"{date_str} 00:00:00",
        end=f"{date_str} 23:00:00",
        freq="h", tz="UTC",
    )
    pred_rows, hourly_est_users = [], []
    for ts in pred_hours:
        u, _ = _lookup(ts.dayofweek, ts.hour)
        hourly_est_users.append(u)
        pred_rows.append({
            "event_hour": ts, "source": params["source"],
            "medium": params["medium"], "device_type": params["device"],
            "os": params["os"], "users": u, "page_views": 0,
            "conversions": 0, "revenue": 0.0,
        })
    pred_df = pd.DataFrame(pred_rows)

    # ── 4. Объединяем: история → прогноз ────────────────────────────────────
    combined = (
        pd.concat([context_df, pred_df], ignore_index=True)
        .sort_values("event_hour")
        .reset_index(drop=True)
    )

    pattern_stats = {
        "total_est_users": sum(hourly_est_users),
        "peak_hour": int(np.argmax(hourly_est_users)),
        "avg_pv_hist": round(global_pv, 1),
        "segment_rows": len(seg_hist),
    }

    return combined, len(pred_df), hourly_est_users, pattern_stats


# ── Бизнес-метрики ────────────────────────────────────────────────────────────
def calculate_business_metrics(
    total_pv: float,
    total_est_users: int,
    params: dict,
    history_df: pd.DataFrame,
) -> dict:
    cpc = params["cpc"]
    is_paid = cpc > 0 and ("cpc" in params["medium"].lower() or "paid" in params["medium"].lower())

    conv_rate = compute_conv_rate(history_df)
    avg_order_value = config.AVG_ORDER_VALUE
    churn = config.MONTHLY_CHURN_RATE
    gross_margin = config.GROSS_MARGIN

    ad_spend = cpc * total_est_users if is_paid else 0.0

    # Конверсии и выручка — только от paid трафика
    pred_pv_paid = total_pv if is_paid else 0.0
    pred_conv = pred_pv_paid * conv_rate
    pred_revenue = pred_conv * avg_order_value

    roi = ((pred_revenue - ad_spend) / ad_spend * 100) if ad_spend > 0 else None
    roas = (pred_revenue / ad_spend) if ad_spend > 0 else None
    cpa = (ad_spend / pred_conv) if (pred_conv > 0 and is_paid) else None
    ltv = avg_order_value * (1.0 / churn) * gross_margin

    return dict(
        conv_rate=conv_rate, ad_spend=ad_spend, pred_conv=pred_conv,
        pred_revenue=pred_revenue, roi=roi, roas=roas, cpa=cpa, ltv=ltv,
    )


# ── Сбор параметров ───────────────────────────────────────────────────────────
def collect_inputs(args) -> dict:
    if args.non_interactive:
        return dict(
            date=args.date, cpc=args.cpc,
            source=args.source, medium=args.medium,
            device=args.device, os=args.os,
            model=args.model, fold=args.fold,
        )

    _header("Прогноз посещаемости и оценка рекламы")
    print()
    print(CYAN("  Введите параметры — модель предскажет трафик на весь день"))
    print(CYAN("  и оценит эффективность рекламной кампании."))
    print(CYAN("  Количество пользователей берётся из исторического паттерна сегмента."))
    print()

    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    _section("1/5  Дата прогноза")
    if args.date:
        date_str = args.date
    else:
        date_str = _ask("Дата прогноза (YYYY-MM-DD)", default=tomorrow)
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(RED("  Неверный формат даты, используем завтра."))
        date_str = tomorrow

    _section("2/5  Источник трафика")
    source = args.source if args.source else _ask("Источник (source)", options=SOURCES, default="google")

    _section("3/5  Канал и реклама")
    medium = args.medium if args.medium else _ask("Канал (medium)", options=MEDIUMS, default="organic")
    cpc = 0.0
    if "cpc" in medium.lower() or "paid" in medium.lower():
        if args.cpc is not None:
            cpc = args.cpc
        else:
            cpc = _ask("Стоимость клика CPC ($)", options=CPCS, default=0.30, cast=float)

    _section("4/5  Устройство и ОС")
    device = args.device if args.device else _ask("Тип устройства", options=DEVICES, default="desktop")
    os = args.os if args.os else _ask("Операционная система", options=OS_LIST, default="Windows")

    _section("5/5  Модель и фолд")
    model = args.model if args.model else _ask("Выберите модель", options=MODELS, default="CatBoost")
    fold = args.fold if args.fold else _ask("Выберите фолд", options=FOLDS, default="5")

    return dict(date=date_str, cpc=cpc, source=source, medium=medium,
                device=device, os=os, model=model, fold=fold)


# ── Вывод результатов ─────────────────────────────────────────────────────────
def print_results(
    params: dict,
    hourly_preds: list[int],
    hourly_est_users: list[int],
    pattern_stats: dict,
    biz: dict,
    model_name: str,
) -> None:
    total_pv = sum(hourly_preds)
    total_est_users = pattern_stats["total_est_users"]

    _header(f"Прогноз на {params['date']} (Модель: {model_name})")

    _section("Параметры сегмента")
    print(f"  Источник/Канал : {params['source']} / {params['medium']}" +
          (f"   CPC = ${params['cpc']:.2f}" if biz["ad_spend"] > 0 else ""))
    print(f"  Устройство     : {params['device']}  /  ОС: {params['os']}")
    print(f"  Пользователей  : ~{total_est_users} за день "
          f"(ист. паттерн по {pattern_stats['segment_rows']} строкам)")

    _section("Прогноз трафика (page_views) и ожидаемые пользователи")
    print(f"  Всего page_views за день : {BOLD(str(total_pv))}")
    print(f"  Среднее page_views/час   : {total_pv // 24}")
    print(f"  Конверсий (оценка)       : ~{biz['pred_conv']:.1f}"
          f"  (conv_rate {biz['conv_rate']*100:.1f}%)")
    print()

    max_pv = max(hourly_preds)   or 1
    max_u = max(hourly_est_users) or 1
    print(f"  {'Час':<6}  {'Прогноз page_views':<32}  {'pv':>5}  {'users':>6}")
    print(f"  {'───':<6}  {'──────────────────────────────────':<32}  {'──':>5}  {'─────':>6}")
    for h, (pv, u) in enumerate(zip(hourly_preds, hourly_est_users)):
        bar = _bar(pv, max_pv, width=32)
        peak = CYAN("◀ пик") if pv == max_pv else " "
        print(f"  {h:02d}:00   {bar}  {pv:>5}  {u:>6} {peak}")

    _section("Бизнес-показатели")
    print(f"  Прогноз выручки : ${biz['pred_revenue']:>9.2f}")
    print(f"  Затраты на рекл.: ${biz['ad_spend']:>9.2f}")
    if biz["roi"] is not None:
        roi_str = f"{biz['roi']:+.1f}%"
        print(f"  ROI             : {GREEN(roi_str) if biz['roi'] > 0 else RED(roi_str)}")
        print(f"  ROAS            : {biz['roas']:.2f}x")
        if biz["cpa"] is not None:
            print(f"  CPA             : ${biz['cpa']:.2f}")
    print(f"  LTV (расч.)     : ${biz['ltv']:.2f}")

    _section("Оценка эффективности рекламы")
    if biz["ad_spend"] == 0:
        print(GREEN("  ✓  ОРГАНИЧЕСКИЙ ТРАФИК — рекламные затраты отсутствуют"))
        print(f"\n  Канал '{params['medium']}' не требует оплаты за клик.")
        print(f"  Прогнозируется {total_pv} просмотров без рекламного бюджета.")
        print(f"  Рекомендация: усиление SEO и контент-маркетинга.")
    elif biz["roi"] > 100:
        print(GREEN(f"  ✓  КАМПАНИЯ ПРИБЫЛЬНА — ROI {biz['roi']:.0f}%"))
        print(f"\n  Каждый $1 рекламных затрат приносит ${1 + biz['roi']/100:.1f} выручки.")
        print(f"  Рекомендация: масштабирование кампании, увеличение бюджета.")
    elif biz["roi"] > 0:
        print(YELLOW(f"  ⚠  РЕКЛАМА ОКУПАЕТСЯ — ROI {biz['roi']:.0f}%"))
        print(f"\n  Кампания прибыльна, но маржа невысокая.")
        cpc = params["cpc"]
        print(f"  Рекомендация: снизить CPC до ${max(0.05, cpc-0.1):.2f}"
              f" или улучшить посадочную страницу.")
    else:
        print(RED(f"  ✗  РЕКЛАМА УБЫТОЧНА — ROI {biz['roi']:.0f}%"))
        print(f"\n  Затраты (${biz['ad_spend']:.2f}) превышают выручку (${biz['pred_revenue']:.2f}).")
        print(f"  Рекомендация: снизить CPC или переключиться на органические каналы.")


# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Прогноз трафика и оценка эффективности рекламы"
    )
    parser.add_argument("--date", default=None, help="Дата YYYY-MM-DD")
    parser.add_argument("--cpc", default=0.30, type=float)
    parser.add_argument("--source", default="google", choices=SOURCES)
    parser.add_argument("--medium", default="organic", choices=MEDIUMS)
    parser.add_argument("--device", default="desktop", choices=DEVICES)
    parser.add_argument("--os", default="Windows", choices=OS_LIST)
    parser.add_argument("--model", default="CatBoost", choices=MODELS)
    parser.add_argument("--fold", default="5")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()

    if args.date:
        args.non_interactive = True

    params = collect_inputs(args)

    # ── Веса модели ───────────────────────────────────────────────────────────
    model_obj = load_model_weights(params["model"], params["cpc"], params["fold"])
    if model_obj is None:
        print(RED("\n  [!] Веса модели не найдены. Запустите main.py для обучения.\n"))
        sys.exit(1)

    # ── Энкодеры (LabelEncoder + seg_stats из обучения) ──────────────────────
    encoders, seg_stats = load_encoders(params["model"], int(params["fold"]), params["cpc"])
    if encoders is None:
        print(RED("\n  [!] Энкодеры не найдены. Переобучите модель (main.py теперь их сохраняет).\n"))
        sys.exit(1)

    # ── Исторические данные для паттерна и лагов ─────────────────────────────
    history_df = load_data(str(config.DATA_PATH))

    # ── Синтетическая история + окно прогноза ────────────────────────────────
    combined_df, n_pred, hourly_est_users, pattern_stats = build_prediction_context(
        params, history_df
    )

    # ── Признаки и предсказание ───────────────────────────────────────────────
    try:
        X, _, _, _, _ = build_feature_matrix(
            combined_df,
            cpc=params["cpc"],
            encoders=encoders,
            seg_stats=seg_stats,
            log_transform=True,
        )
        X_pred = X.iloc[-n_pred:].reset_index(drop=True)

        preds_log = np.maximum(model_obj.predict(X_pred), 0)
        hourly_preds = [max(0, round(float(np.expm1(p)))) for p in preds_log]

    except Exception as e:
        print(RED(f"\n  [!] Ошибка предсказания: {e}\n"))
        sys.exit(1)

    # ── Бизнес-метрики ────────────────────────────────────────────────────────
    biz = calculate_business_metrics(
        total_pv=sum(hourly_preds),
        total_est_users=pattern_stats["total_est_users"],
        params=params,
        history_df=history_df,
    )

    print_results(params, hourly_preds, hourly_est_users, pattern_stats, biz, params["model"])


if __name__ == "__main__":
    main()
