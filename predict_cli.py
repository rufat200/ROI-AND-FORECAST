"""
predict_cli.py — консольный инструмент прогноза трафика и оценки рекламы.
 
Запуск:
    python predict_cli.py
    python predict_cli.py --date 2026-05-25 --users 20 --cpc 0.30
    python predict_cli.py --help
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

def compute_conv_rate():
    df = load_data(str(config.DATA_PATH))

    paid_mask = df['medium'].str.lower().str.contains("cpc|paid|ppc", na=False)
    paid_data = df[paid_mask]
    
    total_users = paid_data["users"].sum()
    total_convs = paid_data["conversions"].sum()
    if total_users == 0:
        return 0.015
        
    return total_convs / total_users
 
W = 64   # ширина вывода
 
def _line(char="─"): print(char * W)
def _header(text): _line("═"); print(f"  {text}"); _line("═")
def _section(text): print(); _line(); print(f"  {text}"); _line()
 
def _bar(value, max_val, width=30, char="█", empty="░"):
    filled = round(value / max_val * width) if max_val > 0 else 0
    return char * filled + empty * (width - filled)
 
def _color(text, code):
    """ANSI цвет. Работает в большинстве терминалов Windows 10+/Linux/Mac."""
    return f"\033[{code}m{text}\033[0m"
 
GREEN = lambda t: _color(t, "32")
RED = lambda t: _color(t, "31")
YELLOW = lambda t: _color(t, "33")
CYAN = lambda t: _color(t, "36")
BOLD = lambda t: _color(t, "1")
 
 
SOURCES = ["google", "(direct)", "<Other>", "shop.googlemerchandisestore.com",
            "(data deleted)"]
MEDIUMS = ["organic", "(none)", "cpc", "referral", "<Other>", "(data deleted)"]
DEVICES = ["desktop", "mobile", "tablet"]
OS_LIST = ["Windows", "Web", "Android", "iOS", "Macintosh", "<Other>"]
MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
FOLDS = [f"{i}" for i in range(1, config.N_SPLITS+1)]
CPCS = [i for i in config.CPC_SEGMENTS]




def build_prediction_context(
    params: dict,
    history_df: pd.DataFrame,
    context_hours: int = 168,
) -> tuple[pd.DataFrame, int]:
    """
    Чтобы лаговые признаки (lag_1, lag_24, rolling_mean_24 и т.д.) были
    вычислены корректно, к синтетическому 24-часовому окну НУЖНО
    добавить реальную историю из датасета.

    Алгоритм:
    1. Берём последние `context_hours` строк реального датасета
       для того же сегмента (source/medium/device/os).
    2. Присоединяем 24-часовое окно прогноза (page_views=0 — заглушка).
    3. build_feature_matrix обработает всё вместе; нам нужны только
       последние 24 строки (индексы [-24:]).

    Возвращает: (combined_df, n_pred_rows)
    """
    date_str   = params["date"]
    hours_24   = pd.date_range(
        start=f"{date_str} 00:00:00",
        end=f"{date_str} 23:00:00",
        freq="h", tz="UTC",
    )

    # Реалистичное почасовое распределение пользователей (пик днём).
    # Нормированный внутридневной паттерн (сумма = 24 → среднее = 1).
    HOURLY_WEIGHT = np.array([
        0.3, 0.2, 0.2, 0.2, 0.2, 0.3,   # 00-05 ночь
        0.5, 0.7, 0.9, 1.1, 1.2, 1.2,   # 06-11 утро
        1.2, 1.2, 1.2, 1.2, 1.2, 1.2,   # 12-17 день
        1.1, 1.0, 0.9, 0.8, 0.6, 0.5,   # 18-23 вечер
    ], dtype=float)
    HOURLY_WEIGHT /= HOURLY_WEIGHT.mean()   # нормируем → сумма ≈ 24

    daily_users    = max(1, params["users"])
    hourly_users_f = daily_users / 24.0 * HOURLY_WEIGHT
    hourly_users   = np.maximum(1, np.round(hourly_users_f)).astype(int)

    pred_df = pd.DataFrame({
        "event_hour":  hours_24,
        "source":      params["source"],
        "medium":      params["medium"],
        "device_type": params["device"],
        "os":          params["os"],
        "users":       hourly_users,
        "page_views":  0,
        "conversions": 0,
        "revenue":     0.0,
    })

    # Берём реальную историю для того же сегмента.
    seg_mask = (
        (history_df["source"]      == params["source"]) &
        (history_df["medium"]      == params["medium"]) &
        (history_df["device_type"] == params["device"]) &
        (history_df["os"]          == params["os"])
    )
    seg_history = history_df[seg_mask].tail(context_hours).copy()

    if len(seg_history) == 0:
        # Если для сегмента нет истории — берём любые строки для контекста лагов.
        print("[warn] Нет истории для сегмента, использую глобальный контекст.")
        seg_history = history_df.tail(context_hours).copy()

    combined = (
        pd.concat([seg_history, pred_df], ignore_index=True)
        .sort_values("event_hour")
        .reset_index(drop=True)
    )
    return combined, len(pred_df)

def calculate_business_metrics(total_pv: float, params: dict):
    users  = params["users"]
    cpc    = params["cpc"]
    is_paid = cpc > 0 and ("cpc" in params["medium"].lower() or "paid" in params["medium"].lower())

    avg_order_value = config.AVG_ORDER_VALUE
    churn           = config.MONTHLY_CHURN_RATE
    gross_margin    = config.GROSS_MARGIN
    conv_rate       = compute_conv_rate()

    ad_spend = cpc * users if is_paid else 0.0

    # Выручка и конверсии — только от платного трафика.
    pred_pv_for_rev = total_pv if is_paid else 0.0
    pred_conv       = pred_pv_for_rev * conv_rate
    pred_revenue    = pred_conv * avg_order_value

    roi  = ((pred_revenue - ad_spend) / ad_spend * 100) if ad_spend > 0 else None
    roas = (pred_revenue / ad_spend)                    if ad_spend > 0 else None
    cpa  = (ad_spend / pred_conv) if (pred_conv > 0 and is_paid) else None
    ltv  = avg_order_value * (1.0 / churn) * gross_margin

    return pred_conv, pred_revenue, ad_spend, roi, roas, cpa, ltv, conv_rate, cpc

def _ask(prompt, options=None, default=None, cast=str):
    """Интерактивный ввод с валидацией."""
    while True:
        if options:
            for i, o in enumerate(options, 1):
                print(f"    {i}. {o}")
            hint = f"[1-{len(options)}]" + (f", Enter={default}" if default else "")
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
                val = cast(raw)
                return val
            except (ValueError, TypeError):
                print(RED(f"  Неверный формат. Попробуйте снова."))
 
 
def collect_inputs(args):
    """
    Возвращает dict с параметрами.
    Если args заданы через CLI — используем их, иначе интерактивный ввод.
    """
    if args.non_interactive:
        return {
            "date": args.date,
            "users": args.users,
            "cpc": args.cpc,
            "source": args.source,
            "medium": args.medium,
            "device": args.device,
            "os": args.os,
            "model": args.model,
        }
 
    _header("Прогноз посещаемости и оценка рекламы")
    print()
    print(CYAN("  Введите параметры — модель предскажет трафик на весь день"))
    print(CYAN("  и оценит эффективность рекламной кампании."))
    print()
 
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
 
    _section("1/6  Дата и масштаб")
    if args.date:
        date_str = args.date
    else:
        date_str = _ask("Дата прогноза (YYYY-MM-DD)", default=tomorrow)
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(RED("  Неверный формат даты, используем завтра.")); date_str = tomorrow
 
    if args.users is not None:
        users = max(1, min(10000, args.users))
    else:
        users = _ask("Ожидаемое число пользователей (users)", default=15, cast=int)
        users = max(1, min(10000, users))
 
    _section("2/6  Источник трафика")
    source = args.source if args.source else _ask("Источник (source)", options=SOURCES, default="google")
 
    _section("3/6  Канал и реклама")
    medium = args.medium if args.medium else _ask("Канал (medium)", options=MEDIUMS, default="organic")
    cpc = 0.30
    if "cpc" in medium.lower() or "paid" in medium.lower():
        if args.cpc is not None:
            cpc = args.cpc
        else:
            cpc = _ask("Стоимость клика CPC ($)", options=CPCS, default=0.30, cast=float)
 
    _section("4/6  Устройство и ОС (операционная система)")
    device = args.device if args.device else _ask("Тип устройства", options=DEVICES, default="desktop")
    os = args.os if args.os else _ask("Операционная система", options=OS_LIST, default="Windows")
 
    _section("5/6  Модель")
    model = args.model if args.model else _ask("Выберите модель", options=MODELS, default="CatBoost")

    _section("6/6 Fold")
    fold = args.fold if args.fold else _ask("Выберите фолд", options=FOLDS, default=5)
 
    return dict(date=date_str, users=users, cpc=cpc,
                source=source, medium=medium, device=device,
                os=os, model=model, fold=fold)
 
 
def load_model_weights(model, cpc, fold):
    ch = f'./model_weights/{model}_fold-{fold}_cpc_{cpc:.2f}.pkl'
    try:
        model = joblib.load(ch)
        return model
    except Exception as e:
        print(RED(f"  Ошибка загрузки {ch}: \n{e}"))
        return None
 
 
def print_results(params: dict, hourly_preds: list, model_name: str) -> None:
    """Форматированный вывод результатов в терминал."""

    total_pv = sum(hourly_preds)
    pred_conv, pred_rev, ad_spend, roi, roas, cpa, ltv, conv_rate, cpc = calculate_business_metrics(total_pv, params)
 
    _header(f"Прогноз на {params['date']} (Модель: {model_name})")

    _section("Параметры сегмента")
    print(f"  Пользователей/день  : {params['users']}")
    print(f"  Источник/Канал     : {params['source']} / {params['medium']}" + 
          (f"   CPC = ${params['cpc']:.2f}" if ad_spend > 0 else ""))
    print(f"  Устройство     : {params['device']}  /  ОС: {params['os']}")

    _section("Прогноз трафика (page_views)")
    print(f"  Всего за день  : {BOLD(str(total_pv))} просмотров")
    print(f"  Среднее/час    : {total_pv // 24}")
    print(f"  Конверсий      : ~{pred_conv:.1f}  (conv_rate {conv_rate*100:.1f}%)")
    print()
 
    max_h = max(hourly_preds) if hourly_preds else 1
    print(f"  {'Час':<6}  {'Модель прогнозирует (pv)':<32}  {'pv':>4}")
    print(f"  {'───':<6}  {'───────────────────────────────':<32}  {'──':>4}")
    for h, val in enumerate(hourly_preds):
        bar = _bar(val, max_h, width=30)
        peak = CYAN("◀ пик") if val == max_h else ""
        print(f"  {h:02d}:00   {bar}  {int(val):>4}  {peak}")
 
    _section("Бизнес-показатели")
    print(f"  Прогноз выручки : ${pred_rev:>9.2f}")
    print(f"  Затраты на рекл.: ${ad_spend:>9.2f}")
    if roi is not None:
        roi_str = f"{roi:+.1f}%"
        print(f"  ROI             : {GREEN(roi_str) if roi > 0 else RED(roi_str)}")
        print(f"  ROAS            : {roas:.2f}x")
        print(f"  CPA             : ${cpa:.2f}")
    print(f"  LTV (расч.)     : ${ltv:.2f}")
 
    _section("Оценка эффективности рекламы")
    if ad_spend == 0:
        print(GREEN("  ✓  ОРГАНИЧЕСКИЙ ТРАФИК — рекламные затраты отсутствуют"))
        print(f"\n  Канал '{params['medium']}' не требует оплаты за клик.")
        print(f"  Прогнозируется {total_pv} просмотров без рекламного бюджета.")
        print(f"  Рекомендация: усиление SEO и контент-маркетинга.")
    elif roi > 100:
        print(GREEN(f"  ✓  КАМПАНИЯ ПРИБЫЛЬНА — ROI {roi:.0f}%"))
        print(f"\n  Каждый $1 рекламных затрат приносит ${1 + roi/100:.1f} выручки.")
        print(f"  Рекомендация: масштабирование кампании, увеличение бюджета.")
    elif roi > 0:
        print(YELLOW(f"  ⚠  РЕКЛАМА ОКУПАЕТСЯ — ROI {roi:.0f}%"))
        print(f"\n  Кампания прибыльна, но маржа невысокая.")
        print(f"  Рекомендация: оптимизация CPC (снизить до ${max(0.05, cpc-0.1):.2f}) или")
        print(f"                улучшение посадочной страницы для роста конверсии.")
    else:
        print(RED(f"  ✗  РЕКЛАМА УБЫТОЧНА — ROI {roi:.0f}%"))
        print(f"\n  Затраты (${ad_spend:.2f}) превышают ожидаемую выручку (${pred_rev:.2f}).")
        print(f"  Рекомендация: снизить CPC или переключиться на органические каналы.")
 
 
# def save_report(params: dict, total_pv: int, hourly: list[float]) -> None:
#     """Сохраняет текстовый отчёт в outputs/."""
#     out = Path("outputs")
#     out.mkdir(exist_ok=True)
 
#     lines = [
#         f"FORECAST REPORT — {params['date']}",
#         "=" * W,
#         f"Дата прогноза : {params['date']}",
#         f"Сегмент       : {params['source']} / {params['medium']} / "
#         f"{params['device']} / {params['os']}",
#         f"Пользователей : {params['users']}",
#         f"CPC           : ${params['cpc']:.2f}",
#         f"Модель        : {params['model']}",
#         "",
#         "Почасовой прогноз:",
#         "-" * W,
#     ]
#     for h, v in enumerate(hourly):
#         lines.append(f"  {h:02d}:00   {int(v):>5} pv")
 
#     is_paid = params["cpc"] > 0 and "cpc" in params["medium"].lower()
#     conv_rate = 0.015
#     convs = total_pv * conv_rate
#     revenue = convs * 50
#     ad_spend = params["cpc"] * params["users"] if is_paid else 0
#     roi = ((revenue - ad_spend) / ad_spend * 100) if ad_spend > 0 else None
 
#     lines += [
#         "",
#         "-" * W,
#         f"  Итого page_views : {total_pv}",
#         f"  Прогноз выручки  : ${revenue:.2f}",
#         f"  Затраты на рекл. : ${ad_spend:.2f}",
#         f"  ROI              : {roi:.1f}%" if roi is not None else "  ROI : N/A (органика)",
#         "",
#         f"  Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
#         "=" * W,
#     ]
 
#     path = out / f"forecast_{params['date']}.txt"
#     path.write_text("\n".join(lines), encoding="utf-8")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Прогноз трафика и оценка эффективности рекламы"
    )
    parser.add_argument("--date", default=None, help="Дата YYYY-MM-DD")
    parser.add_argument("--users", default=None, type=int)
    parser.add_argument("--cpc", default=0.30, type=float)
    parser.add_argument("--source", default="google", choices=SOURCES)
    parser.add_argument("--medium", default="organic", choices=MEDIUMS)
    parser.add_argument("--device", default="desktop", choices=DEVICES)
    parser.add_argument("--os", default="Windows", choices=OS_LIST)
    parser.add_argument("--model", default="CatBoost", choices=MODELS)
    parser.add_argument("--non-interactive", action="store_true",
                        help="Без интерактивных вопросов (все параметры через флаги)")
    parser.add_argument("--fold", default=5, type=int)
    args = parser.parse_args()

    if args.date and args.users:
        args.non_interactive = True

    params = collect_inputs(args)

    # ── Загрузка весов модели ─────────────────────────────────────────────────
    model_obj = load_model_weights(params["model"], params["cpc"], params["fold"])
    if model_obj is None:
        print(RED("\n  [!] КРИТИЧЕСКАЯ ОШИБКА: Веса модели не найдены."))
        print(RED("  Скрипт не может выполнить прогноз без реальной модели. Завершение работы.\n"))
        sys.exit(1)

    # ── Загрузка энкодеров (label encoders + seg_stats из обучения) ───────────
    encoders, seg_stats = load_encoders(params["model"], params["fold"], params["cpc"])
    if encoders is None:
        print(RED("\n  [!] Энкодеры не найдены. Переобучите модель — main.py теперь сохраняет энкодеры."))
        sys.exit(1)

    # ── Загрузка реального датасета для контекста лагов ──────────────────────
    history_df = load_data(str(config.DATA_PATH))

    # ── Построение DataFrame: история + 24 часа прогноза ─────────────────────
    combined_df, n_pred = build_prediction_context(params, history_df)

    try:
        X, _, _, _, _ = build_feature_matrix(
            combined_df,
            cpc=params["cpc"],
            encoders=encoders,     # используем энкодеры из обучения!
            seg_stats=seg_stats,   # используем статистику из обучения!
            log_transform=True,
        )

        # Берём только последние n_pred строк — это и есть прогноз на 24 часа
        X_pred = X.iloc[-n_pred:].reset_index(drop=True)

        preds_log   = model_obj.predict(X_pred)
        preds_log   = np.maximum(preds_log, 0)
        hourly_preds = [max(0, round(float(np.expm1(p)))) for p in preds_log]

    except Exception as e:
        print(RED(f"\n  [!] ОШИБКА ПРЕДСКАЗАНИЯ МОДЕЛИ: {e}"))
        print(RED("  Проверьте пайплайн признаков (features.py). Завершение работы.\n"))
        sys.exit(1)

    print_results(params, hourly_preds, params["model"])
 
 
if __name__ == "__main__":
    main()
